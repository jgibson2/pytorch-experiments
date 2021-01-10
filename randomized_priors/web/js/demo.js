paper.install(window);
window.onload = function() {
    paper.setup('myCanvas');
    // Create a simple drawing tool:
    var tool = new Tool();
    var path;
    var c = document.getElementById("myCanvas");
    var ctx = c.getContext("2d");

    const myOnnxSession = new onnx.InferenceSession();
    myOnnxSession.loadModel("./models/combined_classifier.onnx").then( () => {
        // Define a mousedown and mousedrag handler
        tool.onMouseDown = function(event) {
            path = new Path();
            path.strokeColor = 'white';
            path.strokeWidth = 40;
            path.strokeCap = 'round';
            path.strokeJoin = 'round';
            path.add(event.point);
        }

        tool.onMouseDrag = function(event) {
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
                let inferenceInputs = preprocess(destCtx.getImageData(0, 0, canvas.width, canvas.height), canvas.width, canvas.height);
                let win = window.open();
                let url = canvas.toDataURL();
                win.document.write(
                    '<iframe src="' + url + '" frameborder="0" style="border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;" allowfullscreen></iframe>');
                myOnnxSession.run(inferenceInputs).then((output) => {
                // consume the output
                const outputTensor = output.values().next().value;
                console.log(`model output tensor: ${outputTensor.data}.`);
            });
            }
        });

        $('#clearBtn').click(() => {
            paper.project.activeLayer.removeChildren();
            paper.view.draw();
        });
    });
}

function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 1), [1, 1, height, width]);

  // Normalize 0-255 to 0-1
  ndarray.ops.divseq(dataFromImage, 255.0);

  // Realign imageData from [28*28*4] to the correct dimension [1*1*28*28].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));

  return dataProcessed.data;
}
