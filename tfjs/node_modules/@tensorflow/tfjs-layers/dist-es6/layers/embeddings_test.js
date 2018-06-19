import { tensor3d, test_util } from '@tensorflow/tfjs-core';
import * as tfl from '../index';
import { describeMathCPU } from '../utils/test_utils';
var expectArraysClose = test_util.expectArraysClose;
describeMathCPU('Embedding Layers: Symbolic 1D, 2D & 3D', function () {
    var inputShapes = [[1], [5], [1, 20], [2, 3, 4]];
    var outputDims = [1, 7, 47];
    var batchDim = 17;
    var inputDim = 100;
    var _loop_1 = function (inputShape) {
        var _loop_2 = function (outputDim) {
            var testTitle = "inputShape=" + inputShape + ", outputDim=" + outputDim;
            it(testTitle, function () {
                var embeddingLayer = tfl.layers.embedding({ inputDim: inputDim, outputDim: outputDim });
                var fullInputShape = [batchDim].concat(inputShape);
                var symbolicInput = new tfl.SymbolicTensor('float32', fullInputShape, null, [], null);
                var output = embeddingLayer.apply(symbolicInput);
                var expectedShape = [batchDim].concat(inputShape).concat([outputDim]);
                expect(output.shape).toEqual(expectedShape);
            });
        };
        for (var _i = 0, outputDims_1 = outputDims; _i < outputDims_1.length; _i++) {
            var outputDim = outputDims_1[_i];
            _loop_2(outputDim);
        }
    };
    for (var _i = 0, inputShapes_1 = inputShapes; _i < inputShapes_1.length; _i++) {
        var inputShape = inputShapes_1[_i];
        _loop_1(inputShape);
    }
});
describeMathCPU('Embedding Layers: With explicit inputLength', function () {
    var inputShape = [null, 4, 5];
    var outputDim = 7;
    var inputLengths = [[null, 4, 5], [null, null, 5], [null, null, null]];
    var batchDim = 17;
    var _loop_3 = function (inputLength) {
        var testTitle = "inputLength=" + inputLength;
        it(testTitle, function () {
            var inputDim = 100;
            var embeddingLayer = tfl.layers.embedding({ inputDim: inputDim, outputDim: outputDim, inputLength: inputLength });
            var fullInputShape = [batchDim].concat(inputShape);
            var symbolicInput = new tfl.SymbolicTensor('float32', fullInputShape, null, [], null);
            var output = embeddingLayer.apply(symbolicInput);
            var expectedShape = [batchDim].concat(inputShape).concat([outputDim]);
            expect(output.shape).toEqual(expectedShape);
            expect(output.dtype).toEqual(symbolicInput.dtype);
        });
    };
    for (var _i = 0, inputLengths_1 = inputLengths; _i < inputLengths_1.length; _i++) {
        var inputLength = inputLengths_1[_i];
        _loop_3(inputLength);
    }
});
describeMathCPU('Embedding Layers: Tensor', function () {
    it('check value equality', function () {
        var x = tensor3d([0, 5, 1, 1, 1, 1, 1, 1], [1, 2, 4]);
        var embeddingLayer = tfl.layers.embedding({ inputDim: 6, outputDim: 3, embeddingsInitializer: 'randomUniform' });
        var y = embeddingLayer.apply(x);
        var yExpectedShape = [1, 2, 4, 3];
        expect(y.shape).toEqual(yExpectedShape);
        var weights = embeddingLayer.getWeights()[0];
        expect(embeddingLayer.computeOutputShape(x.shape)).toEqual(yExpectedShape);
        var yData0 = y.slice([0, 0, 0, 0], [1, 1, 1, 3]).dataSync();
        var yData1 = y.slice([0, 0, 1, 0], [1, 1, 1, 3]).dataSync();
        var yData2 = y.slice([0, 0, 2, 0], [1, 1, 1, 3]).dataSync();
        var yData3 = y.slice([0, 0, 3, 0], [1, 1, 1, 3]).dataSync();
        var wData0 = weights.slice([0, 0], [1, 3]).dataSync();
        var wData1 = weights.slice([1, 0], [1, 3]).dataSync();
        var wData5 = weights.slice([5, 0], [1, 3]).dataSync();
        expectArraysClose(yData0, wData0);
        expect(function () {
            expectArraysClose(yData0, wData1);
        }).toThrow();
        expectArraysClose(yData1, wData5);
        expect(function () {
            expectArraysClose(yData1, wData0);
        }).toThrow();
        expectArraysClose(yData2, yData3);
    });
});
//# sourceMappingURL=embeddings_test.js.map