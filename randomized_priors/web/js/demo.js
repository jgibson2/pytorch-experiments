Array.prototype.count = function (val) {
    return this.reduce((count, item) => count + (item == val), 0)
}

window.onload = function () {
    paper.setup('myCanvas');
    // Create a simple drawing tool:
    let tool = new paper.Tool();
    let path;
    let c = document.getElementById("myCanvas");

    let layout = {
        autosize: false,
        width: 400,
        height: 400,
        margin: {
            l: 50,
            r: 50,
            b: 50,
            t: 50,
            pad: 4
        },
        yaxis: {
            title: 'Classifier Votes',
            range: [0, 10]
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
            type: 'bar'
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

    const myOnnxSession = new onnx.InferenceSession({backendHint: "cpu"});
    myOnnxSession.loadModel("./models/combined_classifier.onnx").then(() => {
        // myOnnxSession.loadModel("./models/mnist-8.onnx").then(() => {
        // Define a mousedown and mousedrag handler
        tool.onMouseDown = function (event) {
            path = new paper.Path();
            path.strokeColor = 'white';
            path.strokeWidth = 20;
            path.strokeCap = 'round';
            path.strokeJoin = 'round';
            path.add(event.point);
        }

        tool.onMouseDrag = function (event) {
            path.add(event.point);
        }

        $('#submitBtn').click(() => {
            let canvas = document.createElement('canvas');
            canvas.width = 28;
            canvas.height = 28;

            //grab the context from your destination canvas
            let destCtx = canvas.getContext('2d');
            destCtx.fillStyle = "black";
            destCtx.fillRect(0, 0, canvas.width, canvas.height);
            let destImage = new Image();
            destImage.src = c.toDataURL();

            destImage.onload = () => {
                destCtx.drawImage(destImage, 0, 0, canvas.width, canvas.height);
                const inferenceInput = preprocess(
                    destCtx.getImageData(0, 0, canvas.width, canvas.height).data,
                    canvas.width,
                    canvas.height
                );
                console.log(inferenceInput)
                myOnnxSession.run([inferenceInput]).then((output) => {
                    // consume the output
                    console.log(output)
                    const outputTensor = Array.from(output.values().next().value.data);
                    console.log(`model output tensor: ${outputTensor}.`);
                    // console.log(`best guess: ${outputTensor.data.indexOf(Math.max(...outputTensor))}`);

                    data[0].y = [...Array(10).keys()].map(e => outputTensor.count(e));
                    console.log(data)
                    Plotly.update('chart', data, layout);

                }).catch(err => {
                    console.log(err);
                });
            }
        });

        $('#clearBtn').click(() => {
            paper.project.activeLayer.removeChildren();
            paper.view.draw();
            data[0].y = Array(10).fill(0);
            Plotly.update('chart', data, layout);
        });
    });
}