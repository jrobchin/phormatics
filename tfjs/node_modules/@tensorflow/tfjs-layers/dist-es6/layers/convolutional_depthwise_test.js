import * as tfc from '@tensorflow/tfjs-core';
import { tensor4d } from '@tensorflow/tfjs-core';
import * as tfl from '../index';
import { describeMathCPU, describeMathCPUAndGPU, expectTensorsClose } from '../utils/test_utils';
import { depthwiseConv2d } from './convolutional_depthwise';
describeMathCPUAndGPU('depthwiseConv2d', function () {
    var x4by4Data = [[[
                [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
                [-20, -40, -60, -80]
            ]]];
    var dataFormats = [undefined, 'channelsFirst', 'channelsLast'];
    var paddingModes = [undefined, 'same', 'valid'];
    var stridesArray = [1, 2];
    var depthMultipliers = [1, 2];
    var _loop_1 = function (dataFormat) {
        for (var _i = 0, paddingModes_1 = paddingModes; _i < paddingModes_1.length; _i++) {
            var paddingMode = paddingModes_1[_i];
            var _loop_2 = function (stride) {
                var _loop_3 = function (depthMultiplier) {
                    var testTitle = "stride=" + stride + ", " + paddingMode + ", " +
                        (dataFormat + ", depthMultiplier=" + depthMultiplier);
                    it(testTitle, function () {
                        var x = tensor4d(x4by4Data, [1, 1, 4, 4]);
                        if (dataFormat !== 'channelsFirst') {
                            x = tfc.transpose(x, [0, 2, 3, 1]);
                        }
                        var kernel;
                        if (depthMultiplier === 1) {
                            kernel = tensor4d([1, 0, 0, -1], [2, 2, 1, 1]);
                        }
                        else if (depthMultiplier === 2) {
                            kernel = tensor4d([1, -1, 0, 0, 0, 0, -1, 1], [2, 2, 1, 2]);
                        }
                        var y = depthwiseConv2d(x, kernel, [stride, stride], 'valid', dataFormat);
                        var yExpected;
                        if (stride === 1) {
                            if (depthMultiplier === 1) {
                                yExpected = tensor4d([[[[-30, -30, -30], [50, 90, 130], [30, 30, 30]]]], [1, 1, 3, 3]);
                            }
                            else if (depthMultiplier === 2) {
                                yExpected = tensor4d([[
                                        [[-30, -30, -30], [50, 90, 130], [30, 30, 30]],
                                        [[30, 30, 30], [-50, -90, -130], [-30, -30, -30]]
                                    ]], [1, 2, 3, 3]);
                            }
                        }
                        else if (stride === 2) {
                            if (depthMultiplier === 1) {
                                yExpected = tensor4d([[[[-30, -30], [30, 30]]]], [1, 1, 2, 2]);
                            }
                            else if (depthMultiplier === 2) {
                                yExpected = tensor4d([[[[-30, -30], [30, 30]], [[30, 30], [-30, -30]]]], [1, 2, 2, 2]);
                            }
                        }
                        if (dataFormat !== 'channelsFirst') {
                            yExpected = tfc.transpose(yExpected, [0, 2, 3, 1]);
                        }
                        expectTensorsClose(y, yExpected);
                    });
                };
                for (var _i = 0, depthMultipliers_1 = depthMultipliers; _i < depthMultipliers_1.length; _i++) {
                    var depthMultiplier = depthMultipliers_1[_i];
                    _loop_3(depthMultiplier);
                }
            };
            for (var _a = 0, stridesArray_1 = stridesArray; _a < stridesArray_1.length; _a++) {
                var stride = stridesArray_1[_a];
                _loop_2(stride);
            }
        }
    };
    for (var _i = 0, dataFormats_1 = dataFormats; _i < dataFormats_1.length; _i++) {
        var dataFormat = dataFormats_1[_i];
        _loop_1(dataFormat);
    }
    it('Non-4D kernel leads to exception', function () {
        var x = tfc.zeros([1, 1, 4, 4]);
        expect(function () { return depthwiseConv2d(x, tfc.zeros([1, 2, 2]), [1, 1]); })
            .toThrowError(/.* is required to be 4-D, but is instead 3-D/);
    });
});
describeMathCPU('DepthwiseConv2D-Symbolic', function () {
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var kernelSizes = [2, [2, 2]];
    var depthMultipliers = [1, 3];
    var paddingModes = ['valid', 'same'];
    var _loop_4 = function (dataFormat) {
        var _loop_5 = function (kernelSize) {
            var _loop_6 = function (depthMultiplier) {
                var _loop_7 = function (padding) {
                    var testTitle = "dataFormat=" + dataFormat + ", " +
                        ("kernelSize=" + JSON.stringify(kernelSize) + ", ") +
                        ("depthMultiplier=" + depthMultiplier + ", ") +
                        ("paddingMode=" + padding);
                    it(testTitle, function () {
                        var depthwiseConvLayer = tfl.layers.depthwiseConv2d({ dataFormat: dataFormat, kernelSize: kernelSize, depthMultiplier: depthMultiplier, padding: padding });
                        var inputShape = dataFormat === 'channelsFirst' ? [1, 8, 10, 10] :
                            [1, 10, 10, 8];
                        var symbolicInput = new tfl.SymbolicTensor('float32', inputShape, null, [], null);
                        var symbolicOutput = depthwiseConvLayer.apply(symbolicInput);
                        var outputImageSize = padding === 'valid' ? 9 : 10;
                        var expectedShape;
                        if (dataFormat === 'channelsFirst') {
                            expectedShape =
                                [1, 8 * depthMultiplier, outputImageSize, outputImageSize];
                        }
                        else {
                            expectedShape =
                                [1, outputImageSize, outputImageSize, 8 * depthMultiplier];
                        }
                        expect(symbolicOutput.shape).toEqual(expectedShape);
                    });
                };
                for (var _i = 0, paddingModes_2 = paddingModes; _i < paddingModes_2.length; _i++) {
                    var padding = paddingModes_2[_i];
                    _loop_7(padding);
                }
            };
            for (var _i = 0, depthMultipliers_2 = depthMultipliers; _i < depthMultipliers_2.length; _i++) {
                var depthMultiplier = depthMultipliers_2[_i];
                _loop_6(depthMultiplier);
            }
        };
        for (var _i = 0, kernelSizes_1 = kernelSizes; _i < kernelSizes_1.length; _i++) {
            var kernelSize = kernelSizes_1[_i];
            _loop_5(kernelSize);
        }
    };
    for (var _i = 0, dataFormats_2 = dataFormats; _i < dataFormats_2.length; _i++) {
        var dataFormat = dataFormats_2[_i];
        _loop_4(dataFormat);
    }
    it('Non-4D Array Input leads to exception', function () {
        var depthwiseConvLayer = tfl.layers.depthwiseConv2d({ kernelSize: 2 });
        var symbolicInput = new tfl.SymbolicTensor('float32', [1, 10, 10], null, [], null);
        expect(function () { return depthwiseConvLayer.apply(symbolicInput); })
            .toThrowError(/Inputs to DepthwiseConv2D should have rank 4\. Received .*/);
    });
});
describeMathCPUAndGPU('DepthwiseConv2D-Tensor:', function () {
    var x4by4Data = [[[
                [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
                [-20, -40, -60, -80]
            ]]];
    var depthMultipliers = [1, 2];
    var useBiases = [false];
    var biasInitializers = ['zeros', 'ones'];
    var _loop_8 = function (depthMultiplier) {
        var _loop_9 = function (useBias) {
            var _loop_10 = function (biasInitializer) {
                var testTitle = "channelsFirst, depthMultiplier=" + depthMultiplier + ", " +
                    ("useBias=" + useBias + ", biasInitializer=" + biasInitializer + ", ") +
                    "activation=relu";
                it(testTitle, function () {
                    var x = tensor4d(x4by4Data, [1, 1, 4, 4]);
                    var conv2dLayer = tfl.layers.depthwiseConv2d({
                        kernelSize: [2, 2],
                        depthMultiplier: depthMultiplier,
                        strides: [2, 2],
                        dataFormat: 'channelsFirst',
                        useBias: useBias,
                        depthwiseInitializer: 'ones',
                        biasInitializer: biasInitializer,
                        activation: 'relu'
                    });
                    var y = conv2dLayer.apply(x);
                    var yExpectedShape;
                    var yExpectedData;
                    if (depthMultiplier === 1) {
                        yExpectedShape = [1, 1, 2, 2];
                        yExpectedData = [100, 260, -100, -260];
                    }
                    else if (depthMultiplier === 2) {
                        yExpectedShape = [1, 2, 2, 2];
                        yExpectedData = [100, 260, -100, -260, 100, 260, -100, -260];
                    }
                    if (useBias && biasInitializer === 'ones') {
                        yExpectedData = yExpectedData.map(function (element) { return element + 1; });
                    }
                    yExpectedData =
                        yExpectedData.map(function (element) { return element >= 0 ? element : 0; });
                    var yExpected = tensor4d(yExpectedData, yExpectedShape);
                    expectTensorsClose(y, yExpected);
                });
            };
            for (var _i = 0, biasInitializers_1 = biasInitializers; _i < biasInitializers_1.length; _i++) {
                var biasInitializer = biasInitializers_1[_i];
                _loop_10(biasInitializer);
            }
        };
        for (var _i = 0, useBiases_1 = useBiases; _i < useBiases_1.length; _i++) {
            var useBias = useBiases_1[_i];
            _loop_9(useBias);
        }
    };
    for (var _i = 0, depthMultipliers_3 = depthMultipliers; _i < depthMultipliers_3.length; _i++) {
        var depthMultiplier = depthMultipliers_3[_i];
        _loop_8(depthMultiplier);
    }
    it('channelsLast', function () {
        var x = tfc.transpose(tensor4d(x4by4Data, [1, 1, 4, 4]), [0, 2, 3, 1]);
        var conv2dLayer = tfl.layers.depthwiseConv2d({
            depthMultiplier: 2,
            kernelSize: [2, 2],
            strides: [2, 2],
            dataFormat: 'channelsLast',
            useBias: false,
            depthwiseInitializer: 'ones',
            activation: 'linear'
        });
        var y = conv2dLayer.apply(x);
        var yExpected = tensor4d([100, 100, 260, 260, -100, -100, -260, -260], [1, 2, 2, 2]);
        expectTensorsClose(y, yExpected);
    });
});
//# sourceMappingURL=convolutional_depthwise_test.js.map