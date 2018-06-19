import * as tfc from '@tensorflow/tfjs-core';
import { tensor2d, tensor3d, tensor4d } from '@tensorflow/tfjs-core';
import * as tfl from '../index';
import { SymbolicTensor } from '../types';
import { convOutputLength } from '../utils/conv_utils';
import { describeMathCPUAndGPU, expectTensorsClose } from '../utils/test_utils';
import { pool2d } from './pooling';
describeMathCPUAndGPU('pool2d', function () {
    var x4by4Data = [[[
                [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
                [-20, -40, -60, -80]
            ]]];
    var x5by5Data = [[[
                [0, 1, 3, 5, 7], [0, 2, 4, 6, 8], [0, 0, 0, 0, 0], [0, -1, -3, -5, -7],
                [0, -2, -4, -6, -8]
            ]]];
    var poolModes = [undefined, 'max', 'avg'];
    var dataFormats = [undefined, 'channelsFirst', 'channelsLast'];
    var stridesArray = [1, 2];
    var _loop_1 = function (poolMode) {
        var _loop_2 = function (dataFormat) {
            var _loop_3 = function (stride) {
                var testTitle = "4x4, " + stride + ", same, " + dataFormat + ", " +
                    ("" + poolMode);
                it(testTitle, function () {
                    var x = tensor4d(x4by4Data, [1, 1, 4, 4]);
                    if (dataFormat !== 'channelsFirst') {
                        x = tfc.transpose(x, [0, 2, 3, 1]);
                    }
                    var yExpected;
                    if (poolMode === 'avg') {
                        if (stride === 1) {
                            yExpected = tensor4d([[[
                                        [25, 45, 65, 75], [5, 5, 5, 5], [-25, -45, -65, -75],
                                        [-30, -50, -70, -80]
                                    ]]], [1, 1, 4, 4]);
                        }
                        else {
                            yExpected = tensor4d([[[[25, 65], [-25, -65]]]], [1, 1, 2, 2]);
                        }
                    }
                    else {
                        if (stride === 1) {
                            yExpected = tensor4d([[[
                                        [40, 60, 80, 80], [40, 60, 80, 80], [-10, -30, -50, -70],
                                        [-20, -40, -60, -80]
                                    ]]], [1, 1, 4, 4]);
                        }
                        else if (stride === 2) {
                            yExpected = tensor4d([[[[40, 80], [-10, -50]]]], [1, 1, 2, 2]);
                        }
                    }
                    if (dataFormat !== 'channelsFirst') {
                        yExpected = tfc.transpose(yExpected, [0, 2, 3, 1]);
                    }
                    var y = pool2d(x, [2, 2], [stride, stride], 'same', dataFormat, poolMode);
                    expectTensorsClose(y, yExpected);
                });
            };
            for (var _i = 0, stridesArray_1 = stridesArray; _i < stridesArray_1.length; _i++) {
                var stride = stridesArray_1[_i];
                _loop_3(stride);
            }
        };
        for (var _i = 0, dataFormats_1 = dataFormats; _i < dataFormats_1.length; _i++) {
            var dataFormat = dataFormats_1[_i];
            _loop_2(dataFormat);
        }
    };
    for (var _i = 0, poolModes_1 = poolModes; _i < poolModes_1.length; _i++) {
        var poolMode = poolModes_1[_i];
        _loop_1(poolMode);
    }
    var _loop_4 = function (poolMode) {
        it("5x5, 2, same, CHANNEL_FIRST, " + poolMode, function () {
            var x5by5 = tensor4d(x5by5Data, [1, 1, 5, 5]);
            var yExpected = tensor4d(x4by4Data, [1, 1, 4, 4]);
            if (poolMode === 'avg') {
                yExpected = tensor4d([[[[0.75, 4.5, 7.5], [-0.25, -2, -3.5], [-1, -5, -8]]]], [1, 1, 3, 3]);
            }
            else {
                yExpected =
                    tensor4d([[[[2, 6, 8], [0, 0, 0], [0, -4, -8]]]], [1, 1, 3, 3]);
            }
            var y = pool2d(x5by5, [2, 2], [2, 2], 'same', 'channelsFirst', poolMode);
            expectTensorsClose(y, yExpected);
        });
    };
    for (var _a = 0, poolModes_2 = poolModes; _a < poolModes_2.length; _a++) {
        var poolMode = poolModes_2[_a];
        _loop_4(poolMode);
    }
    var _loop_5 = function (poolMode) {
        it("5x5, 2, valid, CHANNEL_LAST, " + poolMode, function () {
            var x5by5 = tfc.transpose(tensor4d(x5by5Data, [1, 1, 5, 5]), [0, 2, 3, 1]);
            var yExpected;
            if (poolMode === 'avg') {
                yExpected = tensor4d([[[[0.75, 4.5], [-0.25, -2]]]], [1, 1, 2, 2]);
            }
            else {
                yExpected = tensor4d([[[[2, 6], [0, 0]]]], [1, 1, 2, 2]);
            }
            var y = pool2d(x5by5, [2, 2], [2, 2], 'valid', 'channelsLast', poolMode);
            expectTensorsClose(y, tfc.transpose(yExpected, [0, 2, 3, 1]));
        });
    };
    for (var _b = 0, poolModes_3 = poolModes; _b < poolModes_3.length; _b++) {
        var poolMode = poolModes_3[_b];
        _loop_5(poolMode);
    }
});
describe('Pooling Layers 1D: Symbolic', function () {
    var poolSizes = [2, 3];
    var stridesList = [null, 1, 2];
    var poolModes = ['avg', 'max'];
    var paddingModes = [undefined, 'valid', 'same'];
    var _loop_6 = function (poolMode) {
        var _loop_7 = function (paddingMode) {
            var _loop_8 = function (poolSize) {
                var _loop_9 = function (strides) {
                    var testTitle = "poolSize=" + poolSize + ", " +
                        (paddingMode + ", " + poolMode);
                    it(testTitle, function () {
                        var inputLength = 16;
                        var inputNumChannels = 11;
                        var inputBatchSize = 2;
                        var inputShape = [inputBatchSize, inputLength, inputNumChannels];
                        var symbolicInput = new SymbolicTensor('float32', inputShape, null, [], null);
                        var poolConstructor = poolMode === 'avg' ?
                            tfl.layers.averagePooling1d :
                            tfl.layers.maxPooling1d;
                        var poolingLayer = poolConstructor({
                            poolSize: poolSize,
                            strides: strides,
                            padding: paddingMode,
                        });
                        var output = poolingLayer.apply(symbolicInput);
                        var expectedOutputLength = convOutputLength(inputLength, poolSize, paddingMode, strides ? strides : poolSize);
                        var expectedShape = [inputBatchSize, expectedOutputLength, inputNumChannels];
                        expect(output.shape).toEqual(expectedShape);
                        expect(output.dtype).toEqual(symbolicInput.dtype);
                    });
                };
                for (var _i = 0, stridesList_1 = stridesList; _i < stridesList_1.length; _i++) {
                    var strides = stridesList_1[_i];
                    _loop_9(strides);
                }
            };
            for (var _i = 0, poolSizes_1 = poolSizes; _i < poolSizes_1.length; _i++) {
                var poolSize = poolSizes_1[_i];
                _loop_8(poolSize);
            }
        };
        for (var _i = 0, paddingModes_1 = paddingModes; _i < paddingModes_1.length; _i++) {
            var paddingMode = paddingModes_1[_i];
            _loop_7(paddingMode);
        }
    };
    for (var _i = 0, poolModes_4 = poolModes; _i < poolModes_4.length; _i++) {
        var poolMode = poolModes_4[_i];
        _loop_6(poolMode);
    }
});
describeMathCPUAndGPU('Pooling Layers 1D: Tensor', function () {
    var poolModes = ['avg', 'max'];
    var strides = [2, 4];
    var poolSizes = [2, 4];
    var batchSize = 2;
    var _loop_10 = function (poolMode) {
        var _loop_11 = function (stride) {
            var _loop_12 = function (poolSize) {
                var testTitle = "stride=" + stride + ", " + poolMode + ", " +
                    ("poolSize=" + poolSize);
                it(testTitle, function () {
                    var x2by8 = tensor2d([
                        [10, 30, 50, 70, 20, 40, 60, 80],
                        [-10, -30, -50, -70, -20, -40, -60, -80]
                    ]);
                    var x2by8by1 = tfc.expandDims(x2by8, 2);
                    var poolConstructor = poolMode === 'avg' ?
                        tfl.layers.averagePooling1d :
                        tfl.layers.maxPooling1d;
                    var poolingLayer = poolConstructor({
                        poolSize: poolSize,
                        strides: stride,
                        padding: 'valid',
                    });
                    var output = poolingLayer.apply(x2by8by1);
                    var outputLength;
                    var expectedOutputVals;
                    if (poolSize === 2) {
                        if (stride === 2) {
                            outputLength = 4;
                            if (poolMode === 'avg') {
                                expectedOutputVals =
                                    [[[20], [60], [30], [70]], [[-20], [-60], [-30], [-70]]];
                            }
                            else {
                                expectedOutputVals =
                                    [[[30], [70], [40], [80]], [[-10], [-50], [-20], [-60]]];
                            }
                        }
                        else if (stride === 4) {
                            outputLength = 2;
                            if (poolMode === 'avg') {
                                expectedOutputVals = [[[20], [30]], [[-20], [-30]]];
                            }
                            else {
                                expectedOutputVals = [[[30], [40]], [[-10], [-20]]];
                            }
                        }
                    }
                    else if (poolSize === 4) {
                        if (stride === 2) {
                            outputLength = 3;
                            if (poolMode === 'avg') {
                                expectedOutputVals =
                                    [[[40], [45], [50]], [[-40], [-45], [-50]]];
                            }
                            else {
                                expectedOutputVals =
                                    [[[70], [70], [80]], [[-10], [-20], [-20]]];
                            }
                        }
                        else if (stride === 4) {
                            outputLength = 2;
                            if (poolMode === 'avg') {
                                expectedOutputVals = [[[40], [50]], [[-40], [-50]]];
                            }
                            else {
                                expectedOutputVals = [[[70], [80]], [[-10], [-20]]];
                            }
                        }
                    }
                    var expectedShape = [batchSize, outputLength, 1];
                    expectTensorsClose(output, tensor3d(expectedOutputVals, expectedShape));
                });
            };
            for (var _i = 0, poolSizes_2 = poolSizes; _i < poolSizes_2.length; _i++) {
                var poolSize = poolSizes_2[_i];
                _loop_12(poolSize);
            }
        };
        for (var _i = 0, strides_1 = strides; _i < strides_1.length; _i++) {
            var stride = strides_1[_i];
            _loop_11(stride);
        }
    };
    for (var _i = 0, poolModes_5 = poolModes; _i < poolModes_5.length; _i++) {
        var poolMode = poolModes_5[_i];
        _loop_10(poolMode);
    }
});
describe('Pooling Layers 2D: Symbolic', function () {
    var poolSizes = [2, 3];
    var poolModes = ['avg', 'max'];
    var paddingModes = [undefined, 'valid', 'same'];
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var poolSizeIsNumberValues = [false, true];
    var _loop_13 = function (poolMode) {
        var _loop_14 = function (paddingMode) {
            var _loop_15 = function (dataFormat) {
                var _loop_16 = function (poolSize) {
                    var _loop_17 = function (poolSizeIsNumber) {
                        var testTitle = "poolSize=" + poolSize + ", " +
                            (dataFormat + ", " + paddingMode + ", ") +
                            (poolMode + ", ") +
                            ("poollSizeIsNumber=" + poolSizeIsNumber);
                        it(testTitle, function () {
                            var inputShape = dataFormat === 'channelsFirst' ?
                                [2, 16, 11, 9] :
                                [2, 11, 9, 16];
                            var symbolicInput = new SymbolicTensor('float32', inputShape, null, [], null);
                            var poolConstructor = poolMode === 'avg' ?
                                tfl.layers.averagePooling2d :
                                tfl.layers.maxPooling2d;
                            var poolingLayer = poolConstructor({
                                poolSize: poolSizeIsNumber ? poolSize : [poolSize, poolSize],
                                padding: paddingMode,
                                dataFormat: dataFormat,
                            });
                            var output = poolingLayer.apply(symbolicInput);
                            var outputRows = poolSize === 2 ? 5 : 3;
                            if (paddingMode === 'same') {
                                outputRows++;
                            }
                            var outputCols = poolSize === 2 ? 4 : 3;
                            if (paddingMode === 'same' && poolSize === 2) {
                                outputCols++;
                            }
                            var expectedShape;
                            if (dataFormat === 'channelsFirst') {
                                expectedShape = [2, 16, outputRows, outputCols];
                            }
                            else {
                                expectedShape = [2, outputRows, outputCols, 16];
                            }
                            expect(output.shape).toEqual(expectedShape);
                            expect(output.dtype).toEqual(symbolicInput.dtype);
                        });
                    };
                    for (var _i = 0, poolSizeIsNumberValues_1 = poolSizeIsNumberValues; _i < poolSizeIsNumberValues_1.length; _i++) {
                        var poolSizeIsNumber = poolSizeIsNumberValues_1[_i];
                        _loop_17(poolSizeIsNumber);
                    }
                };
                for (var _i = 0, poolSizes_3 = poolSizes; _i < poolSizes_3.length; _i++) {
                    var poolSize = poolSizes_3[_i];
                    _loop_16(poolSize);
                }
            };
            for (var _i = 0, dataFormats_2 = dataFormats; _i < dataFormats_2.length; _i++) {
                var dataFormat = dataFormats_2[_i];
                _loop_15(dataFormat);
            }
        };
        for (var _i = 0, paddingModes_2 = paddingModes; _i < paddingModes_2.length; _i++) {
            var paddingMode = paddingModes_2[_i];
            _loop_14(paddingMode);
        }
    };
    for (var _i = 0, poolModes_6 = poolModes; _i < poolModes_6.length; _i++) {
        var poolMode = poolModes_6[_i];
        _loop_13(poolMode);
    }
});
describeMathCPUAndGPU('Pooling Layers 2D: Tensor', function () {
    var x4by4Data = [10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80];
    var poolModes = ['avg', 'max'];
    var strides = [1, 2];
    var batchSizes = [2, 4];
    var channelsArray = [1, 3];
    var _loop_18 = function (poolMode) {
        var _loop_19 = function (stride) {
            var _loop_20 = function (batchSize) {
                var _loop_21 = function (channels) {
                    var testTitle = "stride=" + stride + ", " + poolMode + ", " +
                        ("batchSize=" + batchSize + ", channels=" + channels);
                    it(testTitle, function () {
                        var xArrayData = [];
                        for (var b = 0; b < batchSize; ++b) {
                            for (var c = 0; c < channels; ++c) {
                                xArrayData = xArrayData.concat(x4by4Data);
                            }
                        }
                        var x4by4 = tensor4d(xArrayData, [batchSize, channels, 4, 4]);
                        var poolConstructor = poolMode === 'avg' ?
                            tfl.layers.averagePooling2d :
                            tfl.layers.maxPooling2d;
                        var poolingLayer = poolConstructor({
                            poolSize: [2, 2],
                            strides: [stride, stride],
                            padding: 'valid',
                            dataFormat: 'channelsFirst',
                        });
                        var output = poolingLayer.apply(x4by4);
                        var expectedShape;
                        var expectedOutputSlice;
                        if (poolMode === 'avg') {
                            if (stride === 1) {
                                expectedShape = [batchSize, channels, 3, 3];
                                expectedOutputSlice = [25, 45, 65, 5, 5, 5, -25, -45, -65];
                            }
                            else if (stride === 2) {
                                expectedShape = [batchSize, channels, 2, 2];
                                expectedOutputSlice = [25, 65, -25, -65];
                            }
                        }
                        else {
                            if (stride === 1) {
                                expectedShape = [batchSize, channels, 3, 3];
                                expectedOutputSlice = [40, 60, 80, 40, 60, 80, -10, -30, -50];
                            }
                            else if (stride === 2) {
                                expectedShape = [batchSize, channels, 2, 2];
                                expectedOutputSlice = [40, 80, -10, -50];
                            }
                        }
                        var expectedOutputArray = [];
                        for (var b = 0; b < batchSize; ++b) {
                            for (var c = 0; c < channels; ++c) {
                                expectedOutputArray =
                                    expectedOutputArray.concat(expectedOutputSlice);
                            }
                        }
                        expectTensorsClose(output, tensor4d(expectedOutputArray, expectedShape));
                    });
                };
                for (var _i = 0, channelsArray_1 = channelsArray; _i < channelsArray_1.length; _i++) {
                    var channels = channelsArray_1[_i];
                    _loop_21(channels);
                }
            };
            for (var _i = 0, batchSizes_1 = batchSizes; _i < batchSizes_1.length; _i++) {
                var batchSize = batchSizes_1[_i];
                _loop_20(batchSize);
            }
        };
        for (var _i = 0, strides_2 = strides; _i < strides_2.length; _i++) {
            var stride = strides_2[_i];
            _loop_19(stride);
        }
    };
    for (var _i = 0, poolModes_7 = poolModes; _i < poolModes_7.length; _i++) {
        var poolMode = poolModes_7[_i];
        _loop_18(poolMode);
    }
});
describe('1D Global pooling Layers: Symbolic', function () {
    var globalPoolingLayers = [tfl.layers.globalAveragePooling1d, tfl.layers.globalMaxPooling1d];
    var _loop_22 = function (globalPoolingLayer) {
        var testTitle = "layer=" + globalPoolingLayer.name;
        it(testTitle, function () {
            var inputShape = [2, 11, 9];
            var symbolicInput = new SymbolicTensor('float32', inputShape, null, [], null);
            var layer = globalPoolingLayer({});
            var output = layer.apply(symbolicInput);
            var expectedShape = [2, 9];
            expect(output.shape).toEqual(expectedShape);
            expect(output.dtype).toEqual(symbolicInput.dtype);
        });
    };
    for (var _i = 0, globalPoolingLayers_1 = globalPoolingLayers; _i < globalPoolingLayers_1.length; _i++) {
        var globalPoolingLayer = globalPoolingLayers_1[_i];
        _loop_22(globalPoolingLayer);
    }
});
describeMathCPUAndGPU('1D Global Pooling Layers: Tensor', function () {
    var x3DimData = [
        [[4, -1], [0, -2], [40, -10], [0, -20]],
        [[-4, 1], [0, 2], [-40, 10], [0, 20]]
    ];
    var globalPoolingLayers = [tfl.layers.globalAveragePooling1d, tfl.layers.globalMaxPooling1d];
    var _loop_23 = function (globalPoolingLayer) {
        var testTitle = "globalPoolingLayer=" + globalPoolingLayer.name;
        it(testTitle, function () {
            var x = tensor3d(x3DimData, [2, 4, 2]);
            var layer = globalPoolingLayer({});
            var output = layer.apply(x);
            var expectedOutput;
            if (globalPoolingLayer === tfl.layers.globalAveragePooling1d) {
                expectedOutput = tensor2d([[11, -8.25], [-11, 8.25]], [2, 2]);
            }
            else {
                expectedOutput = tensor2d([[40, -1], [0, 20]], [2, 2]);
            }
            expectTensorsClose(output, expectedOutput);
        });
    };
    for (var _i = 0, globalPoolingLayers_2 = globalPoolingLayers; _i < globalPoolingLayers_2.length; _i++) {
        var globalPoolingLayer = globalPoolingLayers_2[_i];
        _loop_23(globalPoolingLayer);
    }
});
describe('2D Global pooling Layers: Symbolic', function () {
    var globalPoolingLayers = [tfl.layers.globalAveragePooling2d, tfl.layers.globalMaxPooling2d];
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var _loop_24 = function (globalPoolingLayer) {
        var _loop_25 = function (dataFormat) {
            var testTitle = "layer=" + globalPoolingLayer.name + ", " + dataFormat;
            it(testTitle, function () {
                var inputShape = [2, 16, 11, 9];
                var symbolicInput = new SymbolicTensor('float32', inputShape, null, [], null);
                var layer = globalPoolingLayer({ dataFormat: dataFormat });
                var output = layer.apply(symbolicInput);
                var expectedShape = dataFormat === 'channelsLast' ? [2, 9] : [2, 16];
                expect(output.shape).toEqual(expectedShape);
                expect(output.dtype).toEqual(symbolicInput.dtype);
            });
        };
        for (var _i = 0, dataFormats_3 = dataFormats; _i < dataFormats_3.length; _i++) {
            var dataFormat = dataFormats_3[_i];
            _loop_25(dataFormat);
        }
    };
    for (var _i = 0, globalPoolingLayers_3 = globalPoolingLayers; _i < globalPoolingLayers_3.length; _i++) {
        var globalPoolingLayer = globalPoolingLayers_3[_i];
        _loop_24(globalPoolingLayer);
    }
});
describeMathCPUAndGPU('2D Global Pooling Layers: Tensor', function () {
    var x4DimData = [
        [[[4, -1], [0, -2]], [[40, -10], [0, -20]]],
        [[[4, -1], [0, -2]], [[40, -10], [0, -20]]]
    ];
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var globalPoolingLayers = [tfl.layers.globalAveragePooling2d, tfl.layers.globalMaxPooling2d];
    var _loop_26 = function (globalPoolingLayer) {
        var _loop_27 = function (dataFormat) {
            var testTitle = "globalPoolingLayer=" + globalPoolingLayer.name + ", " + dataFormat;
            it(testTitle, function () {
                var x = tensor4d(x4DimData, [2, 2, 2, 2]);
                var layer = globalPoolingLayer({ dataFormat: dataFormat });
                var output = layer.apply(x);
                var expectedOutput;
                if (globalPoolingLayer === tfl.layers.globalAveragePooling2d) {
                    if (dataFormat === 'channelsFirst') {
                        expectedOutput = tensor2d([[0.25, 2.5], [0.25, 2.5]], [2, 2]);
                    }
                    else {
                        expectedOutput = tensor2d([[11, -8.25], [11, -8.25]], [2, 2]);
                    }
                }
                else {
                    if (dataFormat === 'channelsFirst') {
                        expectedOutput = tensor2d([[4, 40], [4, 40]], [2, 2]);
                    }
                    else {
                        expectedOutput = tensor2d([[40, -1], [40, -1]], [2, 2]);
                    }
                }
                expectTensorsClose(output, expectedOutput);
                var config = layer.getConfig();
                expect(config.dataFormat).toEqual(dataFormat);
            });
        };
        for (var _i = 0, dataFormats_4 = dataFormats; _i < dataFormats_4.length; _i++) {
            var dataFormat = dataFormats_4[_i];
            _loop_27(dataFormat);
        }
    };
    for (var _i = 0, globalPoolingLayers_4 = globalPoolingLayers; _i < globalPoolingLayers_4.length; _i++) {
        var globalPoolingLayer = globalPoolingLayers_4[_i];
        _loop_26(globalPoolingLayer);
    }
});
//# sourceMappingURL=pooling_test.js.map