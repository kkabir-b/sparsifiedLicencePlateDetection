# sparsifiedLicencePlateDetection
Lisence plate detection using a sparsified YOLOv8 Model along with paddleOCR

## Sparsification Explanation
Sparsification is the process of making a NN model more efficient and computationally inexpensive. This work uses a sparsified version of yoloV8 object detection model for efficient object detection of licence platesalong with paddleOCR for licence plate character recognition

### Pruning
Pruning in sparsification involves removing redundant connections within a NN. It works on the assumption that NN by design are overly complicated and have unnecessary connections.
This reduces the number of operations during execution and thus increases efficiency (albeit at the cost of accuracy at some times).

### Quantization
Quantization reduces the precision of weights and activations in a NN (for example changing a fp32 to int8) decreasing the time and memory taken up by the NN.
