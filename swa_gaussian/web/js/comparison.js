Array.prototype.count = function (val) {
    return this.reduce((count, item) => count + (item === val), 0)
}

window.onload = function () {
    paper.setup('myCanvas');
    let path;
    let c = document.getElementById("myCanvas");
    let ctx = c.getContext('2d');
    let predIds = new Map();
    predIds.set(0, '#lastPredStandardClassifier')
    predIds.set(1, '#lastPredRandomizedPriors')
    predIds.set(2, '#lastPredSWAG')

    let layout = {
        autosize: false,
        width: 800,
        height: 400,
        margin: {
            l: 50,
            r: 50,
            b: 50,
            t: 50,
            pad: 4
        },
        yaxis: {
            title: 'Prediction Output',
            range: [0, 1]
        },
        xaxis: {
            title: 'Digit',
            tickvals: [...Array(10).keys()],
            ticktext: [...Array(10).keys()].map(n => n.toString()),
            tickmode: 'array',
            range: [-1, 10]
        },
        title: {
            text:`Classifier Comparison`
        }
    };

    let data = [
        {
            x: [...Array(10).keys()],
            y: Array(10).fill(0),
            error_y: {
                type: 'data', array: Array(10).fill(0), visible: false
            },
            type: 'bar',
            name: 'Standard Network'
        },
        {
            x: [...Array(10).keys()],
            y: Array(10).fill(0),
            error_y: {
                type: 'data', array: Array(10).fill(0), visible: false
            },
            type: 'bar',
            name: 'Randomized Priors'
        },
        {
            x: [...Array(10).keys()],
            y: Array(10).fill(0),
            error_y: {
                type: 'data', array: Array(10).fill(0), visible: false
            },
            type: 'bar',
            name: 'SWA-Gaussian'
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

    let inference = (sess, fn, idx, conf) => {
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
        let found = false;
        for ( let i = 0; i < drawnImageData.data.length; i+=4 )
        {
            let x = Math.floor( i / 4 ) % w;
            let y = Math.floor( i / ( 4 * w ) );

            if ( drawnImageData.data[ i ] > 0 || drawnImageData.data[ i + 1 ] > 0 || drawnImageData.data[ i + 2 ] > 0 )
            {
                found = true;
                xmin = Math.min( xmin, x );
                xmax = Math.max( xmax, x );
                ymin = Math.min( ymin, y );
                ymax = Math.max( ymax, y );
            }
        }
        const cropWidth = xmax - xmin;
        const cropHeight = ymax - ymin;
        if(!found) {
            Plotly.update('chart', data, layout);
            return;
        }
        crop.width = cropWidth;
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
            console.log(output);
            console.log(Object.keys(output));
            let outputName = conf ? "output_mean" : "output";
            const outputTensor = Array.from(output.get(outputName).data);
            // console.log(`model output tensor: ${outputTensor}.`);
            data[idx].y = fn(outputTensor);
            if(conf) {
                const confTensor = Array.from(output.get("output_std").data);
                data[idx].error_y.array = fn(confTensor);
                 data[idx].error_y.visible = true;
            }
            // console.log(`final output: ${data[idx].y}.`)
            $(predIds.get(idx)).text(`${data[idx].y.indexOf(Math.max(...data[idx].y))}`);
            Plotly.update('chart', data, layout);
        }).catch(err => {
            console.log(err);
        });
    };

    let softmax = (o) => {
        let smSum = o.reduce((a,e) => a + Math.exp(e), 0.0);
        return o.map(e => Math.exp(e) / smSum);
    }

    let scale = (o) => {
        let smSum = o.reduce((a,e) => a + e, 0.0);
        return o.map(e => e / smSum);
    }

    let scale_max = (o) => {
        let smMax = o.reduce((a,e) => Math.max(a, e), 1e-8);
        return o.map(e => e / smMax);
    }

    let identity = (o) => {return o;}

    let loadModels = (doInference) => {
        let tool = new paper.Tool();
        const randomizedPriorsOnnxSession = new onnx.InferenceSession({backendHint: "cpu"});
        const swagOnnxSession = new onnx.InferenceSession({backendHint: "cpu"});
        const standardOnnxSession = new onnx.InferenceSession({backendHint: "cpu"});
        randomizedPriorsOnnxSession.loadModel(`./models/combined_classifier_randomized_priors.onnx`).then(() => {
            swagOnnxSession.loadModel(`./models/combined_classifier_swa_gaussian.onnx`).then(() => {
                standardOnnxSession.loadModel(`./models/standard_classifier.onnx`).then(() => {
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
                        inference(standardOnnxSession, scale, 0, false);
                        inference(randomizedPriorsOnnxSession, scale, 1, true);
                        inference(swagOnnxSession, scale, 2, true);
                    }

                    if (doInference) {
                        tool.onMouseUp();
                    }
                });
            });
        });
    }

    $('#clearBtn').click(() => {
        paper.project.activeLayer.removeChildren();
        paper.view.draw();
        data.forEach(d => {d.y = Array(10).fill(0);});
        data.forEach(d => {d.error_y.visible = false});
        Plotly.update('chart', data, layout);
    });

    loadModels( false);
}