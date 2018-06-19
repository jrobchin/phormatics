var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
import { eye, ones, scalar, Tensor, tensor1d, tensor2d, zeros } from '@tensorflow/tfjs-core';
import * as tfl from '../index';
import * as initializers from '../initializers';
import { describeMathCPU, describeMathCPUAndGPU, expectTensorsClose } from '../utils/test_utils';
import { LayerVariable, onesVariable, zerosVariable } from '../variables';
import { execute, FeedDict } from './executor';
import { Container, getSourceInputs, Input, InputSpec, loadWeightsFromJson, loadWeightsFromNamedTensorMap, Node } from './topology';
var LayerForTest = (function (_super) {
    __extends(LayerForTest, _super);
    function LayerForTest(config) {
        return _super.call(this, config) || this;
    }
    LayerForTest.className = 'LayerForTest';
    return LayerForTest;
}(tfl.layers.Layer));
var ContainerForTest = (function (_super) {
    __extends(ContainerForTest, _super);
    function ContainerForTest(config) {
        return _super.call(this, config) || this;
    }
    ContainerForTest.className = 'ContainerForTest';
    return ContainerForTest;
}(Container));
describe('InputSpec', function () {
    it('initializes with expected default values.', function () {
        var inputSpec = new InputSpec({});
        expect(inputSpec.dtype).toBeUndefined();
        expect(inputSpec.shape).toBeUndefined();
        expect(inputSpec.ndim).toBeUndefined();
        expect(inputSpec.maxNDim).toBeUndefined();
        expect(inputSpec.minNDim).toBeUndefined();
        expect(inputSpec.axes).toEqual({});
    });
    it('initializes with inputSpec.ndim = shape.length when shape is specified.', function () {
        var shape = [1, 2, 3];
        var expectedValue = shape.length;
        var inputSpec = new InputSpec({ shape: [1, 2, 3], ndim: -1 });
        expect(inputSpec.ndim).toEqual(expectedValue);
    });
    it('initializes inputSpec.axes when axes specified.', function () {
        var expectedValue = { 1: 2 };
        var inputSpec = new InputSpec({ axes: expectedValue });
        expect(inputSpec.axes).toEqual(expectedValue);
    });
});
describe('Node', function () {
    var outboundLayerName = 'outboundLayer';
    var inboundLayerName = 'inboundLayer';
    var outboundLayer = new LayerForTest({ name: outboundLayerName });
    var inboundLayers = [new LayerForTest({ name: inboundLayerName })];
    var nodeIndices = [0];
    var tensorIndices = [0];
    var inputTensors = [new tfl.SymbolicTensor('float32', [1], null, [], {})];
    var outputTensors = [new tfl.SymbolicTensor('float32', [2, 2], null, [], {})];
    var inputMasks = [zeros([1])];
    var outputMasks = [zeros([1])];
    var inputShapes = [[1]];
    var outputShapes = [[1], [1]];
    var callArgs = { mask: zeros([1]) };
    var node = new Node({
        outboundLayer: outboundLayer,
        inboundLayers: inboundLayers,
        nodeIndices: nodeIndices,
        tensorIndices: tensorIndices,
        inputTensors: inputTensors,
        outputTensors: outputTensors,
        inputMasks: inputMasks,
        outputMasks: outputMasks,
        inputShapes: inputShapes,
        outputShapes: outputShapes
    }, callArgs);
    it('initializes object as expected.', function () {
        expect(node.outboundLayer).toEqual(outboundLayer);
        expect(node.inboundLayers).toEqual(inboundLayers);
        expect(node.nodeIndices).toEqual(nodeIndices);
        expect(node.tensorIndices).toEqual(tensorIndices);
        expect(node.inputTensors).toEqual(inputTensors);
        expect(node.outputTensors).toEqual(outputTensors);
        expect(node.inputMasks).toEqual(inputMasks);
        expect(node.outputMasks).toEqual(outputMasks);
        expect(node.inputShapes).toEqual(inputShapes);
        expect(node.outputShapes).toEqual(outputShapes);
        expect(node.callArgs).toEqual(callArgs);
        expect(inboundLayers[0].outboundNodes).toEqual([node]);
        expect(node.outboundLayer.inboundNodes).toEqual([node]);
    });
    it('generates expected SerializableNodeConfig.', function () {
        var nodeConfig = node.getConfig();
        expect(nodeConfig.outboundLayer).toEqual(outboundLayerName);
        expect(nodeConfig.inboundLayers).toEqual([inboundLayerName]);
        expect(nodeConfig.nodeIndices).toEqual(nodeIndices);
        expect(nodeConfig.tensorIndices).toEqual(tensorIndices);
    });
    it('generates unique IDs', function () {
        var secondNode = new Node({
            outboundLayer: outboundLayer,
            inboundLayers: inboundLayers,
            nodeIndices: nodeIndices,
            tensorIndices: tensorIndices,
            inputTensors: inputTensors,
            outputTensors: outputTensors,
            inputMasks: inputMasks,
            outputMasks: outputMasks,
            inputShapes: inputShapes,
            outputShapes: outputShapes
        }, callArgs);
        expect(secondNode.id).not.toEqual(node.id);
    });
});
describeMathCPU('Layer', function () {
    describe('initialized to its defaults', function () {
        var defaultLayer;
        beforeEach(function () {
            defaultLayer = new LayerForTest({});
        });
        it('has a default layer name of layer_....', function () {
            expect(defaultLayer.name).toMatch(/^layer_.+$/);
        });
        it('has null inputSpecs.', function () {
            expect(defaultLayer.inputSpec).toBeNull();
        });
        it('does not support masking (supportsMasking == false).', function () {
            expect(defaultLayer.supportsMasking).toEqual(false);
        });
        it('is trainable.', function () {
            expect(defaultLayer.trainable).toEqual(true);
        });
        it('has an undefined batchInputShape.', function () {
            expect(defaultLayer.batchInputShape).toBeUndefined();
        });
        it('has an undefined dtype.', function () {
            expect(defaultLayer.dtype).toBeUndefined();
        });
        it('has null initialWeights.', function () {
            expect(defaultLayer.initialWeights).toBeNull();
        });
        it('has an empty inboundNodes list.', function () {
            expect(defaultLayer.inboundNodes).toEqual([]);
        });
        it('has an empty outboundNodes list.', function () {
            expect(defaultLayer.outboundNodes).toEqual([]);
        });
        it('has an empty losses list.', function () {
            expect(defaultLayer.losses).toEqual([]);
        });
        it('has an empty updates list.', function () {
            expect(defaultLayer.updates).toEqual([]);
        });
        it('is not built (built == false).', function () {
            expect(defaultLayer.built).toEqual(false);
        });
        it('has an empty trainableWeights list.', function () {
            expect(defaultLayer.trainableWeights).toEqual([]);
        });
        it('has an empty nonTrainableWeights list.', function () {
            expect(defaultLayer.nonTrainableWeights).toEqual([]);
        });
        it('has an empty weights list.', function () {
            expect(defaultLayer.weights).toEqual([]);
        });
        it('produces a unique ID', function () {
            var secondLayer = new LayerForTest({});
            expect(defaultLayer.id).not.toEqual(secondLayer.id);
        });
        it('stateful is false by default', function () {
            var layer = new LayerForTest({});
            expect(layer.stateful).toBe(false);
        });
        it('returns null if it doesn`t support masking and no mask is passed in.', function () {
            expect(defaultLayer.computeMask([], null)).toBeNull();
        });
        it('throws exception if it doesn`t support masking and a ' +
            'mask is passed in.', function () {
            var mask = ones([1]);
            expect(function () { return defaultLayer.computeMask([], mask); })
                .toThrowError(/does not support masking/);
        });
        it('returns the same mask passed in if it supports masking', function () {
            var mask = ones([1]);
            defaultLayer.supportsMasking = true;
            expect(defaultLayer.computeMask([], mask)).toEqual(mask);
        });
        it('correctly generates a config for serialization', function () {
            var config = defaultLayer.getConfig();
            expect(config.name).toEqual(defaultLayer.name);
            expect(config.trainable).toEqual(defaultLayer.trainable);
            expect(config.batchInputShape).toBeUndefined();
            expect(config.dtype).toBeUndefined();
        });
    });
    describe('A layer with non-default arguments', function () {
        it('initializes layer with given name.', function () {
            var name = 'layer name';
            var layer = new LayerForTest({ name: name });
            expect(layer.name).toMatch(name);
            var config = layer.getConfig();
            expect(config.name).toEqual(layer.name);
        });
        var _loop_1 = function (trainable) {
            it('initializes layer as trainable, if specified.', function () {
                var layer = new LayerForTest({ trainable: trainable });
                expect(layer.trainable).toEqual(trainable);
                var config = layer.getConfig();
                expect(config.trainable).toEqual(layer.trainable);
            });
        };
        for (var _i = 0, _a = [true, false]; _i < _a.length; _i++) {
            var trainable = _a[_i];
            _loop_1(trainable);
        }
        var _loop_2 = function (batchInputShape) {
            it('initializes batchInputShape to layerConfig.batchInputShape.', function () {
                var layer = new LayerForTest({ batchInputShape: batchInputShape });
                expect(layer.batchInputShape).toEqual(batchInputShape);
                var config = layer.getConfig();
                expect(config.batchInputShape).toEqual(layer.batchInputShape);
            });
        };
        for (var _b = 0, _c = [[], [1]]; _b < _c.length; _b++) {
            var batchInputShape = _c[_b];
            _loop_2(batchInputShape);
        }
        it('initializes batchInputShape to layerConfig.batchInputShape even if ' +
            'layerConfig.inputShape is defined.', function () {
            var batchInputShape = [1];
            var inputShape = [2, 3];
            var layer = new LayerForTest({ batchInputShape: batchInputShape, inputShape: inputShape });
            expect(layer.batchInputShape).toEqual(batchInputShape);
        });
        var _loop_3 = function (batchSize, inputShape, expectedBatchInputShape) {
            it('initializes batchInputShape to layerConfig.inputShape.', function () {
                var layer = new LayerForTest({ batchSize: batchSize, inputShape: inputShape });
                expect(layer.batchInputShape).toEqual(expectedBatchInputShape);
            });
        };
        for (var _d = 0, _e = [
            [null, [], [null]], [null, [1], [null, 1]], [3, [], [3]],
            [3, [1], [3, 1]]
        ]; _d < _e.length; _d++) {
            var _f = _e[_d], batchSize = _f[0], inputShape = _f[1], expectedBatchInputShape = _f[2];
            _loop_3(batchSize, inputShape, expectedBatchInputShape);
        }
        it('initializes dtype to float32 if layerConfig.inputShape is set.', function () {
            var layer = new LayerForTest({ inputShape: [] });
            expect(layer.dtype).toEqual('float32');
            var config = layer.getConfig();
            expect(config.dtype).toEqual(layer.dtype);
        });
        it('initializes dtype to float32 if layerConfig.batchInputShape is set.', function () {
            var layer = new LayerForTest({ batchInputShape: [] });
            expect(layer.dtype).toEqual('float32');
        });
        it('initializes initialWeights if present.', function () {
            var weights = [zeros([1])];
            var layer = new LayerForTest({ weights: weights });
            expect(layer.initialWeights).toEqual(weights);
        });
        it('Layer with duplicate weight names throws error', function () {
            var LayerForTest = (function (_super) {
                __extends(LayerForTest, _super);
                function LayerForTest(config) {
                    var _this = _super.call(this, config) || this;
                    _this.addWeight('foo', [1, 2], 'float32', initializers.getInitializer('zeros'));
                    _this.addWeight('foo', [2, 3], 'float32', initializers.getInitializer('zeros'));
                    return _this;
                }
                LayerForTest.className = 'LayerForTest';
                return LayerForTest;
            }(tfl.layers.Layer));
            expect(function () { return new LayerForTest({}); })
                .toThrowError(/[Dd]uplicate weight name/);
        });
    });
    it('can be set to built.', function () {
        var layer = new LayerForTest({});
        layer.built = true;
        expect(layer.built).toEqual(true);
    });
    var trainableWeights = [zerosVariable([1])];
    var nonTrainableWeights = [onesVariable([1])];
    it('can set trainableWeights.', function () {
        var layer = new LayerForTest({});
        layer.trainableWeights = trainableWeights;
        expect(layer.trainableWeights).toEqual(trainableWeights);
    });
    it('doesn\'t return trainableWeights if layer is not trainable, even ' +
        'if they exist', function () {
        var layer = new LayerForTest({ trainable: false });
        layer.trainableWeights = trainableWeights;
        expect(layer.trainableWeights).toEqual([]);
    });
    it('can set nonTrainableWeights.', function () {
        var layer = new LayerForTest({});
        layer.nonTrainableWeights = nonTrainableWeights;
        expect(layer.nonTrainableWeights).toEqual(nonTrainableWeights);
    });
    it('only returns nonTrainableWeights for nonTrainableWeights if the layer ' +
        'is trainable.', function () {
        var layer = new LayerForTest({ trainable: true });
        layer.trainableWeights = trainableWeights;
        layer.nonTrainableWeights = nonTrainableWeights;
        expect(layer.nonTrainableWeights).toEqual(nonTrainableWeights);
    });
    it('concats trainable and nonTrainableWeights for nonTrainableWeights if ' +
        'not trainable.', function () {
        var layer = new LayerForTest({ trainable: false });
        var expectedWeights = trainableWeights.concat(nonTrainableWeights);
        layer.trainableWeights = trainableWeights;
        layer.nonTrainableWeights = nonTrainableWeights;
        expect(layer.nonTrainableWeights).toEqual(expectedWeights);
    });
    var _loop_4 = function (trainable) {
        it('concats trainable and nonTrainableWeights for weights regardless of ' +
            'whether the layer is trainable trainable.', function () {
            var layer = new LayerForTest({ trainable: trainable });
            var expectedWeights = trainableWeights.concat(nonTrainableWeights);
            layer.trainableWeights = trainableWeights;
            layer.nonTrainableWeights = nonTrainableWeights;
            expect(layer.weights).toEqual(expectedWeights);
        });
    };
    for (var _i = 0, _a = [true, false]; _i < _a.length; _i++) {
        var trainable = _a[_i];
        _loop_4(trainable);
    }
    describe('assertInputCompatibility()', function () {
        function runAssert(layer, inputs) {
            layer.assertInputCompatibility(inputs);
        }
        var testInputs = [
            function () { return ones([1]); }, function () { return [ones([1])]; },
            function () { return new tfl.SymbolicTensor('float32', [1], null, [], {}); },
            function () { return [new tfl.SymbolicTensor('float32', [1], null, [], {})]; }
        ];
        var _loop_5 = function (inputs) {
            it('doesn\'t raise an exception if no inputSpec is provided.', function () {
                var layer = new LayerForTest({});
                runAssert(layer, inputs());
            });
            it('doesn\'t raise exception if number of inputs == number of ' +
                'inputSpecs.', function () {
                var inputSpecs = [new InputSpec({})];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if number of inputs != number of inputSpecs.', function () {
                var inputSpecs = [new InputSpec({}), new InputSpec({})];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); })
                    .toThrowError(/expects [0-9]+ inputs/);
            });
            it('doesn\'t raise exception if inputs\' ndim == inputSpecs.ndim.', function () {
                var inputSpecs = [new InputSpec({ ndim: 1 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' ndim != inputSpecs.ndim.', function () {
                var inputSpecs = [new InputSpec({ ndim: 2 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).toThrowError(/expected ndim=/);
            });
            it('doesn\'t raise exception if inputs\' ndim <= inputSpecs.maxNdim.', function () {
                var inputSpecs = [new InputSpec({ maxNDim: 1 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' ndim > inputSpecs.maxNdim.', function () {
                var inputSpecs = [new InputSpec({ maxNDim: 0 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); })
                    .toThrowError(/expected max_ndim=/);
            });
            it('doesn\'t raise exception if inputs\' ndim >= inputSpecs.minNdim.', function () {
                var inputSpecs = [new InputSpec({ minNDim: 1 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' ndim < inputSpecs.minNdim.', function () {
                var inputSpecs = [new InputSpec({ minNDim: 2 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); })
                    .toThrowError(/expected min_ndim=/);
            });
            it('doesn\'t raise exception if inputs\' dtype == inputSpecs.dtype.', function () {
                var inputSpecs = [new InputSpec({ dtype: 'float32' })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('doesn\'t raise exception if inputs\' dimensions == inputSpecs.axes.', function () {
                var inputSpecs = [new InputSpec({ axes: { 0: 1 } })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' dimensions != inputSpecs.axes.', function () {
                var inputSpecs = [new InputSpec({ axes: { 0: 2 } })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).toThrowError(/expected axis/);
            });
            it('throws exception if inputs\' dimensions don\'t have the same ' +
                'number of inputSpecs.axes.', function () {
                var inputSpecs = [new InputSpec({ axes: { 0: 1, 2: 1 } })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); })
                    .toThrowError(/expected axis/);
            });
            it('doesn\'t raise exception if inputs\' shape == inputSpecs.shape.', function () {
                var inputSpecs = [new InputSpec({ shape: [1] })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' shape != inputSpecs.shape.', function () {
                var inputSpecs = [new InputSpec({ shape: [2] })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).toThrowError(/expected shape/);
            });
        };
        for (var _i = 0, testInputs_1 = testInputs; _i < testInputs_1.length; _i++) {
            var inputs = testInputs_1[_i];
            _loop_5(inputs);
        }
    });
    describe('apply() passed 1 SymbolicTensor', function () {
        var firstLayer = new LayerForTest({ name: 'firstLayer' });
        var secondLayer = new LayerForTest({ name: 'secondLayer' });
        var callArgs = { a: 1 };
        var singleSymbolicTensor = new tfl.SymbolicTensor('float32', [1], firstLayer, [], {});
        var returnedTensor = secondLayer.apply(singleSymbolicTensor, callArgs);
        it('returns a SymbolicTensor.', function () {
            expect(returnedTensor instanceof tfl.SymbolicTensor).toBe(true);
        });
        it('returns a SymbolicTensor with a reference to the source layer.', function () {
            expect(returnedTensor.sourceLayer).toEqual(secondLayer);
        });
        it('returns a SymbolicTensor with a reference to the inputs passed ' +
            'to apply().', function () {
            expect(returnedTensor.inputs).toEqual([singleSymbolicTensor]);
            expect(returnedTensor.callArgs).toEqual(callArgs);
        });
        it('returns a SymbolicTensor with nodeIndex and tensorIndex set.', function () {
            expect(returnedTensor.nodeIndex).toBeDefined();
            expect(returnedTensor.tensorIndex).toBeDefined();
        });
        it('returns a SymbolicTensor with the name set.', function () {
            expect(returnedTensor.name).toMatch(/secondLayer/);
        });
        it('is built.', function () {
            expect(secondLayer.built).toBe(true);
        });
    });
    describe('apply() passed >1 SymbolicTensor', function () {
        it('throws an exception for multiple symbolic inputs.', function () {
            var firstLayer = new LayerForTest({ name: 'first layer' });
            var secondLayer = new LayerForTest({ name: 'second layer' });
            var symbolicTensorList = [
                new tfl.SymbolicTensor('float32', [1], firstLayer, [], {}, 'first_symbolic_tensor'),
                new tfl.SymbolicTensor('float32', [1], firstLayer, [], {}, 'second_symbolic_tensor')
            ];
            expect(function () { return secondLayer.apply(symbolicTensorList); }).toThrowError();
        });
    });
    describe('apply() passed SymbolicTensor and Tensor', function () {
        it('throws an exception.', function () {
            var layer = new LayerForTest({});
            var inputs = [
                new tfl.SymbolicTensor('float32', [1], null, [], {}, 'first_symbolic_tensor'),
                ones([1])
            ];
            expect(function () { return layer.apply(inputs); })
                .toThrowError(/must be all SymbolicTensors or all Tensors/);
        });
    });
    it('apply() returns multiple symbolic tensors for multiple ' +
        'output shapes', function () {
        var layer = new LayerForTest({});
        var outputShapes = [[1], [2, 3]];
        var input = new tfl.SymbolicTensor('float32', [1], null, [], {});
        spyOn(layer, 'computeOutputShape').and.callFake(function () {
            return outputShapes;
        });
        var results = layer.apply(input);
        expect(results.length).toEqual(2);
        expect(results.map(function (x) { return x.shape; })).toEqual(outputShapes);
        expect(results.map(function (x) { return x.outputTensorIndex; })).toEqual([0, 1]);
    });
    describe('apply() passed 1+ Tensors', function () {
        it('returns new values for output if the same as the input.', function () {
            var anArray = ones([1]);
            for (var _i = 0, _a = [anArray, [anArray, anArray]]; _i < _a.length; _i++) {
                var inputs = _a[_i];
                var layer = new LayerForTest({});
                var result = layer.apply(inputs);
                expect(result instanceof Tensor || (result[0] instanceof Tensor))
                    .toBe(true);
                expect(layer.built).toBe(true);
                if (result instanceof Array) {
                    var inputArray = inputs;
                    for (var i = 0; i < result.length; i++) {
                        expectTensorsClose(result[i], inputArray[i]);
                    }
                }
                else {
                    expectTensorsClose(result, inputs);
                }
                expect(result === inputs).toBe(false);
            }
        });
    });
    describe('initialized with weights at construction time', function () {
        it('sets those weights after calling apply().', function () {
            var initialWeights = eye(2);
            var arrayInput = zeros([1]);
            var symbolicInput = new tfl.SymbolicTensor('float32', [1], null, [], {});
            var _loop_6 = function (inputs) {
                var layer = new LayerForTest({ weights: [initialWeights] });
                spyOn(layer, 'build').and.callFake(function () {
                    layer.built = true;
                    layer.trainableWeights = [new LayerVariable(zeros([2, 2]))];
                });
                expect(layer.weights.length).toEqual(0);
                layer.apply(inputs);
                expect(layer.weights.length).toEqual(1);
                expectTensorsClose(layer.weights[0].read(), initialWeights);
            };
            for (var _i = 0, _a = [arrayInput, symbolicInput]; _i < _a.length; _i++) {
                var inputs = _a[_i];
                _loop_6(inputs);
            }
        });
    });
    describe('apply() (nodes)', function () {
        it('doesn\'t change inboundNodes or outboundNodes when called with ' +
            'concrete input', function () {
            var layer = new LayerForTest({});
            expect(layer.inboundNodes.length).toEqual(0);
            expect(layer.outboundNodes.length).toEqual(0);
            layer.apply(eye(1));
            expect(layer.inboundNodes.length).toEqual(0);
            expect(layer.outboundNodes.length).toEqual(0);
        });
        it('changes inboundNodes and outboundNodes when called with ' +
            'symbolic input', function () {
            var layer = new LayerForTest({});
            var input = new tfl.SymbolicTensor('float32', [1], null, [], {});
            expect(layer.inboundNodes.length).toEqual(0);
            expect(layer.outboundNodes.length).toEqual(0);
            layer.apply(input);
            expect(layer.inboundNodes.length).toEqual(1);
            expect(layer.outboundNodes.length).toEqual(0);
            expect(layer.inboundNodes[0].outboundLayer).toEqual(layer);
        });
        it('updates inbound and outboundNodes when there are multiple layers', function () {
            var firstLayer = new LayerForTest({ name: 'first_layer' });
            var secondLayer = new LayerForTest({ name: 'second_layer' });
            var initialInput = new tfl.SymbolicTensor('float32', [1], null, [], {});
            var firstOutput = firstLayer.apply(initialInput);
            secondLayer.apply(firstOutput);
            expect(firstLayer.inboundNodes.length).toEqual(1);
            expect(firstLayer.outboundNodes.length).toEqual(1);
            expect(secondLayer.inboundNodes.length).toEqual(1);
            expect(secondLayer.outboundNodes.length).toEqual(0);
            expect(firstLayer.outboundNodes[0].outboundLayer).toEqual(secondLayer);
        });
    });
    describe('setWeights', function () {
        it('throws exception if weights are not the same length ' +
            'as existing weights', function () {
            var layer = new LayerForTest({});
            layer.trainableWeights = [new LayerVariable(zeros([2, 2]))];
            var onesTensor = ones([1]);
            expect(function () { return layer.setWeights([
                onesTensor, onesTensor
            ]); }).toThrowError(/with a weight list of length/);
        });
        it('throws exception if weights are not the same shape ' +
            'as existing weights', function () {
            var layer = new LayerForTest({});
            var onesTensor = ones([1]);
            layer.trainableWeights = [new LayerVariable(zeros([2, 2]))];
            expect(function () { return layer.setWeights([onesTensor]); })
                .toThrowError(/not compatible with provided weight shape/);
        });
        it('updates weights.', function () {
            var layer = new LayerForTest({});
            var onesTensor = ones([1]);
            layer.trainableWeights = [new LayerVariable(zeros([1]))];
            layer.setWeights([onesTensor]);
            expectTensorsClose(layer.trainableWeights[0].read(), onesTensor);
        });
    });
    describe('computeOutputShape()', function () {
        it('returns the inputShape in the base class', function () {
            var layer = new LayerForTest({});
            var shape = [1];
            expect(layer.computeOutputShape(shape)).toEqual(shape);
        });
    });
    describe('input and output properties: ', function () {
        var input;
        var layer;
        var output;
        beforeEach(function () {
            input =
                new tfl.SymbolicTensor('float32', [1], null, [], {}, 'firstInput');
            layer = new LayerForTest({});
            output = layer.apply(input);
        });
        it('input retrieves layer\'s inputs.', function () {
            expect(layer.input).toEqual(input);
        });
        it('input retrieves layer\'s outputs.', function () {
            expect(layer.output).toEqual(output);
        });
        it('input throws exception if there is more than one input', function () {
            var secondInput = new tfl.SymbolicTensor('float32', [1], null, [], {}, 'secondInput');
            layer.apply(secondInput);
            expect(function () { return layer.input; }).toThrowError(/"layer input" is ill-defined/);
        });
        it('output throws exception if there is more than one output', function () {
            var secondInput = new tfl.SymbolicTensor('float32', [1], null, [], {}, 'secondInput');
            layer.apply(secondInput);
            expect(function () { return layer.output; }).toThrowError(/"layer output" is ill-defined/);
        });
    });
    describe('getInputAt and getOutputAt: ', function () {
        var input;
        var layer;
        var output;
        beforeEach(function () {
            input =
                new tfl.SymbolicTensor('float32', [1], null, [], {}, 'firstInput');
            layer = new LayerForTest({});
            output = layer.apply(input);
        });
        it('getInputAt() retrieves layer\'s inputs.', function () {
            expect(layer.getInputAt(0)).toEqual(input);
        });
        it('getOutputAt() retrieves layer\'s outputs.', function () {
            expect(layer.getOutputAt(0)).toEqual(output);
        });
        it('getInputAt() throws exception ask for incorrect index.', function () {
            expect(function () { return layer.getInputAt(1); })
                .toThrowError(/Asked to get input at node 1, but/);
        });
        it('getOutputAt() throws exception ask for incorrect index.', function () {
            expect(function () { return layer.getOutputAt(1); })
                .toThrowError(/Asked to get output at node 1, but/);
        });
    });
});
describeMathCPU('InputLayer', function () {
    it('when initialized to its defaults throws an exception', function () {
        expect(function () { return tfl.layers.inputLayer({}); })
            .toThrowError(/InputLayer should be passed either/);
    });
    describe('initialized with only an inputShape', function () {
        var inputShape = [1];
        var inputLayer = tfl.layers.inputLayer({ inputShape: inputShape });
        it('is not trainable.', function () {
            expect(inputLayer.trainable).toBe(false);
        });
        it('is built.', function () {
            expect(inputLayer.built).toBe(true);
        });
        it('is not sparse.', function () {
            expect(inputLayer.sparse).toBe(false);
        });
        it('automatically assigns a name.', function () {
            expect(inputLayer.name).toMatch(/^input.*$/);
        });
        it('creates a batchInputShape of [null].concat(inputShape).', function () {
            expect(inputLayer.batchInputShape).toEqual([null].concat(inputShape));
        });
        it('has no outboundNodes', function () {
            expect(inputLayer.outboundNodes.length).toEqual(0);
        });
        it('has one inboundNode', function () {
            expect(inputLayer.inboundNodes.length).toEqual(1);
        });
        describe('creates an inbound Node', function () {
            var inboundNode = inputLayer.inboundNodes[0];
            it('with no inboundLayers, nodeIndices, or tensorIndices', function () {
                expect(inboundNode.inboundLayers.length).toEqual(0);
                expect(inboundNode.nodeIndices.length).toEqual(0);
                expect(inboundNode.tensorIndices.length).toEqual(0);
            });
            it('with [null] inputMasks and outputMasks', function () {
                expect(inboundNode.inputMasks).toEqual([null]);
                expect(inboundNode.outputMasks).toEqual([null]);
            });
            it('with equal inputShapes and outputShapes', function () {
                expect(inboundNode.inputShapes).toEqual(inboundNode.outputShapes);
                expect(inboundNode.inputShapes).toEqual([[null].concat(inputShape)]);
            });
            describe('with a SymbolicTensor', function () {
                var symbolicTensor = inboundNode.inputTensors[0];
                it('that is defined.', function () {
                    expect(symbolicTensor instanceof tfl.SymbolicTensor).toBe(true);
                });
                it('assigned to both the input and outputTensors.', function () {
                    expect(inboundNode.inputTensors.length).toEqual(1);
                    expect(inboundNode.outputTensors.length).toEqual(1);
                    expect(inboundNode.inputTensors).toEqual(inboundNode.outputTensors);
                });
                it('with a node and tensorIndex of 0.', function () {
                    expect(symbolicTensor.nodeIndex).toEqual(0);
                    expect(symbolicTensor.tensorIndex).toEqual(0);
                });
                it('with a sourceLayer of the inputLayer.', function () {
                    expect(symbolicTensor.sourceLayer).toEqual(inputLayer);
                });
                it('with a name matching the inputLayer name.', function () {
                    expect(symbolicTensor.name).toEqual(inputLayer.name);
                });
                it('with a dtype equal to the inputLayer.', function () {
                    expect(symbolicTensor.dtype).toEqual(inputLayer.dtype);
                });
                it('with a shape matching the inputLayer.batchInputShape', function () {
                    expect(symbolicTensor.shape).toEqual(inputLayer.batchInputShape);
                });
            });
        });
    });
    it('throws an exception if both inputShape and batchInputShape ' +
        'are specified during initialization.', function () {
        expect(function () { return tfl.layers.inputLayer({ inputShape: [1], batchInputShape: [1] }); })
            .toThrowError(/Only provide the inputShape OR batchInputShape/);
    });
    var _loop_7 = function (batchSize) {
        it('initializes with batchSize when inputShape specified', function () {
            var inputShape = [1];
            var inputLayer = tfl.layers.inputLayer({ inputShape: inputShape, batchSize: batchSize });
            expect(inputLayer.batchInputShape).toEqual([
                batchSize
            ].concat(inputShape));
        });
    };
    for (var _i = 0, _a = [null, 5]; _i < _a.length; _i++) {
        var batchSize = _a[_i];
        _loop_7(batchSize);
    }
    it('initializes with batchInputShape if specified.', function () {
        var batchInputShape = [1, 2];
        var inputLayer = tfl.layers.inputLayer({ batchInputShape: batchInputShape });
        expect(inputLayer.batchInputShape).toEqual(batchInputShape);
    });
    it('initializes with batchInputShape if null specified for the batch size.', function () {
        var batchInputShape = [1, 2];
        var inputLayer = tfl.layers.inputLayer({ batchInputShape: batchInputShape });
        expect(inputLayer.batchInputShape).toEqual(batchInputShape);
    });
    it('throws exception if batchSize and batchInputShape are specified.', function () {
        expect(function () { return tfl.layers.inputLayer({ batchInputShape: [1], batchSize: 5 }); })
            .toThrowError(/Cannot specify batchSize if batchInputShape/);
    });
    var _loop_8 = function (sparse) {
        it('uses config.sparse during initialization.', function () {
            var inputLayer = tfl.layers.inputLayer({ inputShape: [1], sparse: sparse });
            expect(inputLayer.sparse).toEqual(sparse);
        });
    };
    for (var _b = 0, _c = [true, false]; _b < _c.length; _b++) {
        var sparse = _c[_b];
        _loop_8(sparse);
    }
    it('use config.dtype during initialization.', function () {
        var dtype = 'float32';
        var inputLayer = tfl.layers.inputLayer({ inputShape: [1], dtype: dtype });
        expect(inputLayer.dtype).toEqual(dtype);
    });
    it('use config.name during initialization.', function () {
        var name = 'abc';
        var inputLayer = tfl.layers.inputLayer({ inputShape: [1], name: name });
        expect(inputLayer.name).toEqual(name);
    });
    it('throws an exception if apply() is called with any input.', function () {
        var inputLayer = tfl.layers.inputLayer({ inputShape: [1] });
        var symbolicTensor = new tfl.SymbolicTensor('float32', [2], null, [], {});
        expect(function () { return inputLayer.apply(symbolicTensor); })
            .toThrowError(/Cannot pass any input to an InputLayer's apply/);
    });
    it('throws an exception if its inputs differ in shape to what it ' +
        'was initialized to.', function () {
        var inputLayer = tfl.layers.inputLayer({ inputShape: [1] });
        var inputs = ones([2, 2]);
        expect(function () { return inputLayer.apply(inputs); }).toThrowError();
    });
    it('returns a serializable config.', function () {
        var batchInputShape = [1];
        var dtype = 'float32';
        var sparse = true;
        var name = 'my_name';
        var inputLayer = tfl.layers.inputLayer({ batchInputShape: batchInputShape, dtype: dtype, sparse: sparse, name: name });
        expect(inputLayer.getConfig())
            .toEqual({ batchInputShape: batchInputShape, dtype: dtype, sparse: sparse, name: name });
    });
});
describe('Input()', function () {
    it('throws an exception if neither shape nor batchShape are specified', function () {
        expect(function () { return tfl.layers.input({}); })
            .toThrowError(/Please provide to Input either/);
    });
    var shape = [1];
    var batchShape = [2, 2];
    var name = 'abc';
    var dtype = 'float32';
    it('returns an initialized symbolicTensor given a shape.', function () {
        var symbolicTensor = tfl.layers.input({ shape: shape, name: name, dtype: dtype });
        expect(symbolicTensor instanceof tfl.SymbolicTensor).toBe(true);
        expect(symbolicTensor.shape).toEqual([null].concat(shape));
        expect(symbolicTensor.name).toMatch(/abc/);
        expect(symbolicTensor.dtype).toEqual(dtype);
    });
    it('returns a SymbolicTensor given a batchShape', function () {
        var symbolicTensor = tfl.layers.input({ batchShape: batchShape });
        expect(symbolicTensor.shape).toEqual(batchShape);
    });
    it('throws exception if both shape and batchShape are specified.', function () {
        expect(function () { return tfl.layers.input({ shape: shape, batchShape: batchShape }); })
            .toThrowError(/Please provide either a `shape`/);
    });
    it('produces output that can feed into a Layer.', function () {
        var inputTensor = Input({ shape: shape, name: name });
        var otherLayer = new LayerForTest({ name: 'firstLayer' });
        var output = otherLayer.apply(inputTensor);
        expect(output instanceof tfl.SymbolicTensor).toBe(true);
        expect(output.name).toEqual('firstLayer/firstLayer');
    });
});
describeMathCPUAndGPU('Container.fromConfig', function () {
    it('creates a minimal Container from simplest config', function () {
        var config = {
            name: 'test',
            layers: [],
            inputLayers: [],
            outputLayers: []
        };
        var container = Container.fromConfig(ContainerForTest, config);
        expect(container.name).toEqual('test');
    });
    it('creates a simple network', function () {
        var config = {
            inputLayers: [['input_2', 0, 0]],
            layers: [
                {
                    className: 'InputLayer',
                    config: {
                        batchInputShape: [null, 32],
                        dtype: 'float32',
                        name: 'input_2',
                        sparse: false
                    },
                    inboundNodes: [],
                    name: 'input_2'
                },
                {
                    className: 'Dense',
                    config: {
                        activation: 'linear',
                        activityRegularizer: null,
                        biasConstraint: null,
                        biasInitializer: { className: 'Zeros', config: {} },
                        biasRegularizer: null,
                        kernelConstraint: null,
                        kernelInitializer: {
                            className: 'VarianceScaling',
                            config: {
                                distribution: 'uniform',
                                mode: 'fanAvg',
                                scale: 1.0,
                                seed: null
                            }
                        },
                        kernelRegularizer: null,
                        name: 'dense_2',
                        trainable: null,
                        units: 32,
                        use_bias: true
                    },
                    inboundNodes: [[['input_2', 0, 0, {}]]],
                    name: 'dense_2'
                }
            ],
            name: 'test',
            outputLayers: [['dense_2', 0, 0]]
        };
        var container = Container.fromConfig(ContainerForTest, config);
        expect(container.name).toEqual('test');
        var allZeros = zeros([1, 32]);
        expectTensorsClose(container.apply(allZeros), allZeros);
    });
});
describeMathCPUAndGPU('Container', function () {
    var inputLayerName = 'inputLayerName';
    var layerName = 'layerName';
    var containerName = 'simpleContainer';
    var inputTensor;
    var layer;
    var output;
    var simpleContainer;
    beforeEach(function () {
        inputTensor = Input({ shape: [1], name: inputLayerName, dtype: 'float32' });
        layer = new LayerForTest({ name: layerName });
        output = layer.apply(inputTensor);
        simpleContainer = new ContainerForTest({ inputs: [inputTensor], outputs: [output], name: containerName });
    });
    it('initializes with no inputs or outputs and a default name', function () {
        var container = new ContainerForTest({ inputs: [], outputs: [] });
        expect(container.name).toMatch(/^container.+$/);
    });
    it('initializes with no inputs or outputs and a given name', function () {
        var name = 'xyz';
        var container = new ContainerForTest({ inputs: [], outputs: [], name: name });
        expect(container.name).toMatch(name);
    });
    it('throws an exception if same input provided twice', function () {
        var makeContainer = function () {
            new ContainerForTest({ inputs: [inputTensor, inputTensor], outputs: [] });
        };
        expect(makeContainer).toThrowError(/inputs.*redundant/);
    });
    it('throws an exception if graph is disconnected', function () {
        var makeContainer = function () {
            new ContainerForTest({ inputs: [], outputs: [output] });
        };
        expect(makeContainer).toThrowError(/disconnected/);
    });
    it('creates inputLayers', function () {
        expect(simpleContainer.inputLayers).toEqual([inputTensor.sourceLayer]);
    });
    it('creates outputLayers', function () {
        expect(simpleContainer.outputLayers).toEqual([layer]);
    });
    it('creates inputNames', function () {
        expect(simpleContainer.inputNames).toEqual([inputLayerName]);
    });
    it('creates outputNames', function () {
        expect(simpleContainer.outputNames).toEqual([layerName]);
    });
    it('throws exception if given a non-input layer as input', function () {
        var makeContainer = function () {
            new ContainerForTest({ inputs: [output], outputs: [] });
        };
        expect(makeContainer).toThrowError(/must be InputLayer objects/);
    });
    it('creates layers for simplest case', function () {
        expect(simpleContainer.layers).toEqual([inputTensor.sourceLayer, layer]);
    });
    it('creates layers when multiple layers specified', function () {
        var layer1 = new LayerForTest({ name: 'layer1' });
        var layer2 = new LayerForTest({ name: 'layer2' });
        var output = layer2.apply(layer1.apply(inputTensor));
        var container = new ContainerForTest({ inputs: [inputTensor], outputs: [output] });
        expect(container.layers).toEqual([inputTensor.sourceLayer, layer1, layer2]);
    });
    it('correctly creates model with shared subgraphs.', function () {
        var layerA = new LayerForTest({ name: 'A' });
        var layerB = new LayerForTest({ name: 'B' });
        var layerC = new LayerForTest({ name: 'C' });
        var layerX = new LayerForTest({ name: 'X' });
        var aOutput = layerA.apply(inputTensor);
        var output1 = layerC.apply(layerB.apply(aOutput));
        var output2 = layerC.apply(layerB.apply(layerX.apply(aOutput)));
        var container = new ContainerForTest({ inputs: [inputTensor], outputs: [output1, output2] });
        var compareFunction = function (a, b) {
            if (a.name < b.name) {
                return -1;
            }
            else if (a.name > b.name) {
                return 1;
            }
            else {
                return 0;
            }
        };
        var sortedLayers = container.layers.slice().sort(compareFunction);
        var expectedSortedLayers = [
            inputTensor.sourceLayer, layerA, layerB, layerC, layerX
        ].sort(compareFunction);
        expect(sortedLayers).toEqual(expectedSortedLayers);
    });
    it('throws exception if multiple layers have the same name', function () {
        var name = 'abc';
        var layer1 = new LayerForTest({ name: name });
        var layer2 = new LayerForTest({ name: name });
        var output = layer2.apply(layer1.apply(inputTensor));
        var makeContainer = function () {
            new ContainerForTest({ inputs: [inputTensor], outputs: [output] });
        };
        expect(makeContainer).toThrowError(/layer names should be unique/);
    });
    it('weights gets all weights.', function () {
        var inputShape = [1, 6];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.dense({ units: 2, useBias: false });
        var layer2 = tfl.layers.dense({ units: 1, useBias: true });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        expect(container.weights.length).toEqual(3);
        expect(container.weights[0].name).toEqual(layer1.weights[0].name);
        expect(container.weights[1].name).toEqual(layer2.weights[0].name);
        expect(container.weights[2].name).toEqual(layer2.weights[1].name);
    });
    it('trainableWeights and nonTrainableWeights.', function () {
        var inputShape = [1, 6];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.dense({ units: 2, useBias: false });
        var layer2 = tfl.layers.dense({ units: 1, useBias: true });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        expect(container.trainableWeights.length).toEqual(3);
        expect(container.trainableWeights[0].name).toEqual(layer1.weights[0].name);
        expect(container.trainableWeights[1].name).toEqual(layer2.weights[0].name);
        expect(container.trainableWeights[2].name).toEqual(layer2.weights[1].name);
        expect(container.nonTrainableWeights.length).toEqual(0);
    });
    it('call() executes all layers.', function () {
        var inputShape = [1, 6];
        var finalShape = [3, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ name: 'layer1', targetShape: [2, 3] });
        var layer2 = tfl.layers.reshape({ name: 'layer2', targetShape: finalShape });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        var result = container.call(ones([1, 1, 6]), {});
        var resultShape = [1].concat(finalShape);
        expectTensorsClose(result[0], ones(resultShape));
    });
    it('apply() executes all layers with concrete tensors.', function () {
        var inputShape = [1, 6];
        var finalShape = [3, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ name: 'layer1', targetShape: [2, 3] });
        var layer2 = tfl.layers.reshape({ name: 'layer2', targetShape: finalShape });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        var result = container.apply(ones([1, 1, 6]));
        var resultShape = [1].concat(finalShape);
        expectTensorsClose(result, ones(resultShape));
    });
    it('apply() executes all layers with symbolic tensors.', function () {
        var inputShape = [1, 6];
        var finalShape = [3, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ name: 'layer1', targetShape: [2, 3] });
        var layer2 = tfl.layers.reshape({ name: 'layer2', targetShape: finalShape });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        var newInput = tfl.layers.input({ shape: [1, 6] });
        var symbolicResult = container.apply(newInput);
        expect(symbolicResult instanceof tfl.SymbolicTensor).toEqual(true);
        var concreteResult = execute(symbolicResult, new FeedDict([{ key: newInput, value: ones([1, 1, 6]) }]));
        var resultShape = [1].concat(finalShape);
        expectTensorsClose(concreteResult, ones(resultShape));
    });
    it('computeOutputShape() computes the correct outputShape', function () {
        var inputShape = [2, 3];
        var finalShape = [3, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer = tfl.layers.reshape({ targetShape: finalShape });
        var output = layer.apply(inputLayer);
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        expect(container.computeOutputShape([1].concat(inputShape))).toEqual([
            1
        ].concat(finalShape));
    });
    it('trainableWeights is initially an empty Array', function () {
        expect(simpleContainer.trainableWeights).toEqual([]);
    });
    it('trainableWeights tracks only trainable weights', function () {
        var inputShape = [2, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ targetShape: [4], name: 'reshapeLayer' });
        var layer1Output = layer1.apply(inputLayer);
        var layer2 = tfl.layers.dense({ units: 2, useBias: false, name: 'denseLayer' });
        var layer2Output = layer2.apply(layer1Output);
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [layer2Output] });
        expect(container.trainableWeights.length).toEqual(1);
    });
    it('stateful is initially false', function () {
        expect(simpleContainer.stateful).toEqual(false);
    });
    function createSimpleTwoLayerContainer() {
        var inputShape = [2, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ targetShape: [4], name: 'reshapeLayer' });
        var layer1Output = layer1.apply(inputLayer);
        var layer2 = tfl.layers.dense({ units: 2, useBias: false, name: 'denseLayer' });
        var layer2Output = layer2.apply(layer1Output);
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [layer2Output] });
        return [container, [container.inputLayers[0], layer1, layer2]];
    }
    it('getLayer works by name', function () {
        var _a = createSimpleTwoLayerContainer(), container = _a[0], layers = _a[1];
        expect(container.getLayer(layers[0].name)).toEqual(layers[0]);
        expect(container.getLayer(layers[1].name)).toEqual(layers[1]);
        expect(container.getLayer(layers[2].name)).toEqual(layers[2]);
    });
    it('getLayer works by index', function () {
        var _a = createSimpleTwoLayerContainer(), container = _a[0], layers = _a[1];
        expect(container.getLayer(null, 0)).toEqual(layers[0]);
        expect(container.getLayer(null, 1)).toEqual(layers[1]);
        expect(container.getLayer(null, 2)).toEqual(layers[2]);
    });
    it('getLayer throws error for nonexistent layer name', function () {
        var _a = createSimpleTwoLayerContainer(), container = _a[0], layers = _a[1];
        expect(function () { return container.getLayer(layers[0].name + '_suffixToMakeLayerNameNonexistent'); })
            .toThrowError(/No such layer/);
    });
    it('getLayer throws error for index out of bound', function () {
        var container = createSimpleTwoLayerContainer()[0];
        expect(function () { return container.getLayer(null, 3); }).toThrowError(/only has 3 layer/);
    });
    it('getLayer throws error when neither name or index is specified', function () {
        var container = createSimpleTwoLayerContainer()[0];
        expect(function () { return container.getLayer(); })
            .toThrowError(/Provide either a layer name or layer index/);
    });
});
describeMathCPUAndGPU('Container.calculateLosses', function () {
    function createSimpleOneLayerContainer(useRegularizers) {
        var inputShape = [2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var kernelRegularizer = useRegularizers ? tfl.regularizers.l1({ l1: 2 }) : null;
        var biasRegularizer = useRegularizers ? tfl.regularizers.l2({ l2: 3 }) : null;
        var denseLayer = tfl.layers.dense({
            units: 2,
            kernelInitializer: 'ones',
            biasInitializer: 'ones',
            kernelRegularizer: kernelRegularizer,
            biasRegularizer: biasRegularizer,
            name: 'denseLayer'
        });
        var layer2Output = denseLayer.apply(inputLayer);
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [layer2Output] });
        return [container, [denseLayer]];
    }
    it('L1 and L2', function () {
        var container = createSimpleOneLayerContainer(true)[0];
        var losses = container.calculateLosses();
        expect(losses.length).toEqual(2);
        expectTensorsClose(losses[0], scalar(2 * (1 + 1 + 1 + 1)));
        expectTensorsClose(losses[1], scalar(3 * (1 + 1)));
    });
    it('No regularizers', function () {
        var container = createSimpleOneLayerContainer(false)[0];
        var losses = container.calculateLosses();
        expect(losses.length).toEqual(0);
    });
});
describe('getSourceInputs()', function () {
    it('returns the single source input', function () {
        var inputTensor = tfl.layers.input({ shape: [1] });
        var layer1 = new LayerForTest({ name: 'layer1' });
        var layer2 = new LayerForTest({ name: 'layer2' });
        var output = layer2.apply(layer1.apply(inputTensor));
        expect(getSourceInputs(output)).toEqual([inputTensor]);
    });
    it('returns all inputs', function () {
        var input1 = tfl.layers.input({ shape: [1], name: 'input1' });
        var input2 = tfl.layers.input({ shape: [1], name: 'input2' });
        var layer = new LayerForTest({});
        var output1 = layer.apply(input1);
        var output2 = layer.apply(input2);
        expect(getSourceInputs(output1)).toEqual([input1]);
        expect(getSourceInputs(output2)).toEqual([input2]);
    });
});
describeMathCPUAndGPU('loadWeightsFromJson', function () {
    var inputTensor = tfl.layers.input({ shape: [3], name: 'inputLayer', dtype: 'float32' });
    it('One layer', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer': [
                    {
                        'name': 'denseLayer/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                    {
                        'name': 'denseLayer/bias:0',
                        'dtype': 'float32',
                        'shape': [2],
                        'value': [-0.1, -0.2],
                    },
                ],
            },
        };
        loadWeightsFromJson(weightsJSON, [denseLayer]);
        expectTensorsClose(denseLayer.apply(tensor2d([[1, 1, 1]], [1, 3])), tensor2d([[0.8, 1.0]], [1, 2]));
    });
    it('Two layers', function () {
        var denseLayer1 = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer1' });
        var denseLayer2 = tfl.layers.dense({ units: 1, useBias: false, name: 'denseLayer2' });
        denseLayer2.apply(denseLayer1.apply(inputTensor));
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer1': [
                    {
                        'name': 'denseLayer1/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                    {
                        'name': 'denseLayer1/bias:0',
                        'dtype': 'float32',
                        'shape': [2],
                        'value': [-0.1, -0.2],
                    },
                ],
                'denseLayer2': [
                    {
                        'name': 'denseLayer2/kernel:0',
                        'dtype': 'float32',
                        'shape': [2, 1],
                        'value': [[1.2], [1.3]],
                    },
                ],
            },
        };
        loadWeightsFromJson(weightsJSON, [denseLayer1, denseLayer2]);
        expectTensorsClose(denseLayer2.apply(denseLayer1.apply(tensor2d([[1, 1, 1]], [1, 3]))), tensor2d([[2.26]], [1, 1]));
    });
    it('Missing weights for a layer', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {},
        };
        expect(function () {
            loadWeightsFromJson(weightsJSON, [denseLayer]);
        })
            .toThrowError(/Layer.*denseLayer.*expects 2 weight.*but.*have 0 element.*/);
    });
    it('Missing a single weight', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer': [
                    {
                        'name': 'denseLayer1/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                    {
                        'name': 'denseLayer1/bias:0',
                        'dtype': 'float32',
                        'shape': [1],
                        'value': [-0.1],
                    },
                ],
            }
        };
        expect(function () {
            loadWeightsFromJson(weightsJSON, [denseLayer]);
        }).toThrowError(/Shape mismatch.*\[2\] vs\. \[1\].*/);
    });
    it('Shape mismatch in a single weight', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer': [
                    {
                        'name': 'denseLayer1/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                ],
            }
        };
        expect(function () {
            loadWeightsFromJson(weightsJSON, [denseLayer]);
        })
            .toThrowError(/Layer.*denseLayer.*expects 2 weight.*but.*have 1 element.*/);
    });
    it('skipMismatch=true tolerates a single missing weight', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer': [
                    {
                        'name': 'denseLayer1/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                ],
            }
        };
        spyOn(console, 'warn');
        loadWeightsFromJson(weightsJSON, [denseLayer], true);
        expect(console.warn).toHaveBeenCalled();
        expectTensorsClose(denseLayer.apply(tensor2d([[1, 1, 1]], [1, 3])), tensor2d([[0.9, 1.2]], [1, 2]));
    });
});
describeMathCPUAndGPU('loadWeightsFromNamedTensorMap', function () {
    var inputTensor = tfl.layers.input({ shape: [3], name: 'inputLayer', dtype: 'float32' });
    it('One layer', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'dense_layer' });
        denseLayer.apply(inputTensor);
        var namedWeightsMap = {};
        namedWeightsMap[denseLayer.weights[0].originalName] =
            tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        namedWeightsMap[denseLayer.weights[1].originalName] = tensor1d([10, 20]);
        loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer]);
        expectTensorsClose(denseLayer.weights[0].read(), tensor2d([1, 2, 3, 4, 5, 6], [3, 2]));
        expectTensorsClose(denseLayer.weights[1].read(), tensor1d([10, 20]));
    });
    it('Unset weights leads to error', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'dense_layer' });
        denseLayer.apply(inputTensor);
        var namedWeightsMap = {};
        namedWeightsMap[denseLayer.weights[0].originalName] =
            tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        expect(function () { return loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer]); })
            .toThrowError(/1 of 2 weights are not set: .*bias.*/);
    });
});
//# sourceMappingURL=topology_test.js.map