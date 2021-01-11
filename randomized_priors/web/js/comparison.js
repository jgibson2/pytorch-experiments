Array.prototype.count = function (val) {
    return this.reduce((count, item) => count + (item == val), 0)
}

window.onload = function () {
    paper.setup('myCanvas');
    // Create a simple drawing tool:
    let tool = new paper.Tool();
    let path;
    let c = document.getElementById("myCanvas");
    let ctx = c.getContext('2d');

    let layout = {
        autosize: false,
        width: 600,
        height: 400,
        margin: {
            l: 50,
            r: 50,
            b: 50,
            t: 50,
            pad: 4
        },
        yaxis: {
            title: 'Confidence',
            range: [0, 1]
        },
        xaxis: {
            title: 'Digit',
            tickvals: [...Array(10).keys()],
            ticktext: [...Array(10).keys()].map(n => n.toString()),
            tickmode: 'array',
            range: [-1, 10]
        }
    };

    let data = [
        {
            x: [...Array(10).keys()],
            y: Array(10).fill(0),
            type: 'bar',
            name: 'Randomized Priors'
        },
        {
            x: [...Array(10).keys()],
            y: Array(10).fill(0),
            type: 'bar',
            name: 'Standard Network'
        }
    ];
    Plotly.newPlot('chart', data, layout);

    let preprocess = (data, width, height) => {
        const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
        const dataProcessed = ndarray(new Float32Array(width * height), [1, 1, width, height]);
        // Normalize 0-255 to 0-1
        ndarray.ops.divseq(dataFromImage, 255.0);
        // Realign imageData from [28*28*4] to the correct dimension [1*1*28*28].
        ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
        // dataProcessed.transpose(0,1,3,2)
        return new onnx.Tensor(dataProcessed.data, 'float32', [1, 1, width, height]);
    }

    let inference = (sess, fn, i) => {
        let crop = document.createElement('canvas');
        let cropCtx = crop.getContext('2d');
        let drawnImageData = ctx.getImageData( 0, 0, ctx.canvas.width, ctx.canvas.height );

        let xmin = ctx.canvas.width - 1;
        let xmax = 0;
        let ymin = ctx.canvas.height - 1;
        let ymax = 0;
        let w = ctx.canvas.width;
        let h = ctx.canvas.height;

        let canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;

        // Find bounding rect of drawing
        for ( let i = 0; i < drawnImageData.data.length; i+=4 )
        {
            let x = Math.floor( i / 4 ) % w;
            let y = Math.floor( i / ( 4 * w ) );

            if ( drawnImageData.data[ i ] > 0 || drawnImageData.data[ i + 1 ] > 0 || drawnImageData.data[ i + 2 ] > 0 )
            {
                xmin = Math.min( xmin, x );
                xmax = Math.max( xmax, x );
                ymin = Math.min( ymin, y );
                ymax = Math.max( ymax, y );
            }
        }
        const cropWidth = xmax - xmin;
        const cropHeight = ymax - ymin;
        crop.width = cropWidth
        crop.height = cropHeight;
        cropCtx.drawImage(c, xmin, ymin, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight)
        const aspectRatio = cropHeight / cropWidth;
        const pad = 2;
        let dx; let dy; let dw; let dh;
        if(aspectRatio > 1.0) {
            // height > width
            dh = canvas.height - (2 * pad);
            dw = Math.round((canvas.width - (2 * pad)) / aspectRatio);
            dy = pad;
            dx = pad + Math.round(((canvas.width - (2 * pad)) - dw) / 2.0);
        } else {
            // height < width
            dw = canvas.width - (2 * pad);
            dh = Math.round((canvas.height - (2 * pad)) * aspectRatio);
            dx = pad;
            dy = pad + Math.round(((canvas.height - (2 * pad)) - dw) / 2.0);
        }

        //grab the context from your destination canvas
        let destCtx = canvas.getContext('2d');
        destCtx.fillStyle = "black";
        destCtx.fillRect(0, 0, canvas.width, canvas.height);
        destCtx.drawImage(crop, 0, 0, crop.width, crop.height, dx, dy, dw, dh);
        $('#lastInput').attr('src', canvas.toDataURL());
        const inferenceInput = preprocess(
            destCtx.getImageData(0, 0, canvas.width, canvas.height).data,
            28,
            28
        );
        sess.run([inferenceInput]).then((output) => {
            // consume the output
            const outputTensor = Array.from(output.values().next().value.data);
            console.log(`model output tensor: ${outputTensor}.`);
            data[i].y = fn(outputTensor);
            $('#lastPred').text(`Last prediction: ${data[i].y.indexOf(Math.max(...data[i].y))}`);
            Plotly.update('chart', data, layout);
        }).catch(err => {
            console.log(err);
        });
    };

    let vote = (o) => {
        return [...Array(10).keys()].map(e => {
            return o.count(e) / o.length
        });
    }

    let softmax = (o) => {
        let smSum = o.reduce((a,e) => a + Math.exp(e), 0.0);
        return o.map(e => Math.exp(e) / smSum);
    }

    let identity = (o) => {return o;}

    const posteriorOnnxSession = new onnx.InferenceSession({backendHint: "cpu"});
    const mnistOnnxSession = new onnx.InferenceSession({backendHint: "cpu"});
//    posteriorOnnxSession.loadModel("./models/voting_classifier.onnx").then(() => {
    posteriorOnnxSession.loadModel("./models/combined_classifier.onnx").then(() => {
        mnistOnnxSession.loadModel("./models/mnist-8.onnx").then(() => {
            tool.onMouseDown = function (event) {
                path = new paper.Path();
                path.strokeColor = 'white';
                path.strokeWidth = 30;
                path.strokeCap = 'round';
                path.strokeJoin = 'round';
                path.add(event.point);
            }

            tool.onMouseDrag = function (event) {
                path.add(event.point);
            }

            tool.onMouseUp = () => {
                inference(mnistOnnxSession, softmax, 1);
                inference(posteriorOnnxSession, identity, 0);
            }

            $('#clearBtn').click(() => {
                paper.project.activeLayer.removeChildren();
                paper.view.draw();
                data.forEach(d => {d.y = Array(10).fill(0);});
                Plotly.update('chart', data, layout);
            });
        });
    });
}