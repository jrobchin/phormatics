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
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
import * as tfc from '@tensorflow/tfjs-core';
import { scalar, tensor2d, tensor3d, tensor4d } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import * as tfl from '../index';
import * as metrics from '../metrics';
import { modelFromJSON } from '../models';
import { convertPythonicToTs, convertTsToPythonic } from '../utils/serialization_utils';
import { describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose } from '../utils/test_utils';
import { rnn, RNNCell } from './recurrent';
function rnnStepForTest(inputs, states) {
    var mean = tfc.mean(inputs);
    var newStates = states.map(function (state) { return K.scalarPlusArray(mean, state); });
    var output = tfc.neg(newStates[0]);
    return [output, newStates];
}
describeMathCPUAndGPU('rnn', function () {
    it('Simple step function: 3D inputs, 1 state', function () {
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        var initialStates = [tfc.zeros([2, 4])];
        var rnnOutputs = rnn(rnnStepForTest, inputs, initialStates);
        var lastOutput = rnnOutputs[0];
        var outputs = rnnOutputs[1];
        var newStates = rnnOutputs[2];
        expectTensorsClose(lastOutput, tensor2d([
            [-57.75, -57.75, -57.75, -57.75],
            [-57.75, -57.75, -57.75, -57.75]
        ], [2, 4]));
        expectTensorsClose(outputs, tensor3d([
            [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
            ],
            [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
            ]
        ], [2, 3, 4]));
        expect(newStates.length).toEqual(1);
        expectTensorsClose(newStates[0], tensor2d([[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]], [2, 4]));
    });
    it('Simple step function: 3D inputs, 2 states', function () {
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        var initialStates = [tfc.zeros([2, 4]), tfc.ones([2, 3])];
        var rnnOutputs = rnn(rnnStepForTest, inputs, initialStates);
        var lastOutput = rnnOutputs[0];
        var outputs = rnnOutputs[1];
        var newStates = rnnOutputs[2];
        expectTensorsClose(lastOutput, tensor2d([
            [-57.75, -57.75, -57.75, -57.75],
            [-57.75, -57.75, -57.75, -57.75]
        ], [2, 4]));
        expectTensorsClose(outputs, tensor3d([
            [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
            ],
            [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
            ]
        ], [2, 3, 4]));
        expect(newStates.length).toEqual(2);
        expectTensorsClose(newStates[0], tensor2d([[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]], [2, 4]));
        expectTensorsClose(newStates[1], tensor2d([[58.75, 58.75, 58.75], [58.75, 58.75, 58.75]], [2, 3]));
    });
    it('Simple step function: 4D inputs, 2 states', function () {
        var inputs = tensor4d([
            [[[1], [2]], [[3], [4]], [[5], [6]]],
            [[[10], [20]], [[30], [40]], [[50], [60]]]
        ], [2, 3, 2, 1]);
        var initialStates = [tfc.zeros([2, 4]), tfc.ones([2, 3])];
        var rnnOutputs = rnn(rnnStepForTest, inputs, initialStates);
        var lastOutput = rnnOutputs[0];
        var outputs = rnnOutputs[1];
        var newStates = rnnOutputs[2];
        expectTensorsClose(lastOutput, tensor2d([
            [-57.75, -57.75, -57.75, -57.75],
            [-57.75, -57.75, -57.75, -57.75]
        ], [2, 4]));
        expectTensorsClose(outputs, tensor3d([
            [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
            ],
            [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
            ]
        ], [2, 3, 4]));
        expect(newStates.length).toEqual(2);
        expectTensorsClose(newStates[0], tensor2d([[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]], [2, 4]));
        expectTensorsClose(newStates[1], tensor2d([[58.75, 58.75, 58.75], [58.75, 58.75, 58.75]], [2, 3]));
    });
    it('Using inputs <3D leads to ValueError', function () {
        var inputs = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var initialStates = [tfc.zeros([4]), tfc.ones([3])];
        expect(function () { return rnn(rnnStepForTest, inputs, initialStates); }).toThrowError();
    });
});
var RNNCellForTest = (function (_super) {
    __extends(RNNCellForTest, _super);
    function RNNCellForTest(stateSizes) {
        var _this = _super.call(this, {}) || this;
        _this.stateSize = stateSizes;
        return _this;
    }
    RNNCellForTest.prototype.call = function (inputs, kwargs) {
        inputs = inputs;
        var dataInputs = inputs[0];
        var states = inputs.slice(1);
        var mean = tfc.mean(dataInputs);
        var newStates = states.map(function (state) { return K.scalarPlusArray(mean, state); });
        var output = tfc.neg(newStates[0]);
        return [output].concat(newStates);
    };
    RNNCellForTest.className = 'RNNCellForTest';
    return RNNCellForTest;
}(RNNCell));
describeMathCPU('RNN-Layer', function () {
    it('constructor: only cell', function () {
        var cell = new RNNCellForTest(5);
        var rnn = tfl.layers.rnn({ cell: cell });
        expect(rnn.returnSequences).toEqual(false);
        expect(rnn.returnState).toEqual(false);
        expect(rnn.goBackwards).toEqual(false);
    });
    it('constructor: cell and custom options', function () {
        var cell = new RNNCellForTest(5);
        var rnn = tfl.layers.rnn({
            cell: cell,
            returnSequences: true,
            returnState: true,
            goBackwards: true
        });
        expect(rnn.returnSequences).toEqual(true);
        expect(rnn.returnState).toEqual(true);
        expect(rnn.goBackwards).toEqual(true);
    });
    it('computeOutputShape: 1 state, returnSequences=false, returnState=false', function () {
        var cell = new RNNCellForTest(5);
        var rnn = tfl.layers.rnn({ cell: cell });
        var inputShape = [4, 3, 2];
        expect(rnn.computeOutputShape(inputShape)).toEqual([4, 5]);
    });
    it('computeOutputShape: 1 state, returnSequences=true, returnState=false', function () {
        var cell = new RNNCellForTest([5, 6]);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true });
        var inputShape = [4, 3, 2];
        expect(rnn.computeOutputShape(inputShape)).toEqual([4, 3, 5]);
    });
    it('computeOutputShape: 1 state, returnSequences=true, returnState=true', function () {
        var cell = new RNNCellForTest(6);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true, returnState: true });
        var inputShape = [4, 3, 2];
        expect(rnn.computeOutputShape(inputShape)).toEqual([[4, 3, 6], [4, 6]]);
    });
    it('computeOutputShape: 2 states, returnSequences=true, returnState=true', function () {
        var cell = new RNNCellForTest([5, 6]);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true, returnState: true });
        var inputShape = [4, 3, 2];
        expect(rnn.computeOutputShape(inputShape)).toEqual([
            [4, 3, 5], [4, 5], [4, 6]
        ]);
    });
    it('apply: Symbolic: 1 state, returnSequences=false, returnState=false', function () {
        var cell = new RNNCellForTest(6);
        var rnn = tfl.layers.rnn({ cell: cell });
        var input = new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
        var output = rnn.apply(input);
        expect(output.shape).toEqual([16, 6]);
    });
    it('apply: Symbolic: 1 state, returnSequences=true, returnState=false', function () {
        var cell = new RNNCellForTest(6);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true });
        var input = new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
        var output = rnn.apply(input);
        expect(output.shape).toEqual([16, 10, 6]);
    });
    it('apply: Symbolic: 1 state, returnSequences=true, returnState=true', function () {
        var cell = new RNNCellForTest(6);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true, returnState: true });
        var input = new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
        var output = rnn.apply(input);
        expect(output.length).toEqual(2);
        expect(output[0].shape).toEqual([16, 10, 6]);
        expect(output[1].shape).toEqual([16, 6]);
    });
    it('apply: Symbolic: 1 state, returnSequences=false, returnState=true', function () {
        var cell = new RNNCellForTest(6);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: false, returnState: true });
        var input = new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
        var output = rnn.apply(input);
        expect(output.length).toEqual(2);
        expect(output[0].shape).toEqual([16, 6]);
        expect(output[1].shape).toEqual([16, 6]);
    });
    it('apply: Symbolic: 2 states, returnSequences=true, returnState=true', function () {
        var cell = new RNNCellForTest([5, 6]);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true, returnState: true });
        var input = new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
        var output = rnn.apply(input);
        expect(output.length).toEqual(3);
        expect(output[0].shape).toEqual([16, 10, 5]);
        expect(output[1].shape).toEqual([16, 5]);
        expect(output[2].shape).toEqual([16, 6]);
    });
});
describeMathCPUAndGPU('RNN-Layer-Math', function () {
    it('getInitialState: 1 state', function () {
        var cell = new RNNCellForTest(5);
        var inputs = tfc.zeros([4, 3, 2]);
        var rnn = tfl.layers.rnn({ cell: cell });
        var initialStates = rnn.getInitialState(inputs);
        expect(initialStates.length).toEqual(1);
        expectTensorsClose(initialStates[0], tfc.zeros([4, 5]));
    });
    it('getInitialState: 2 states', function () {
        var cell = new RNNCellForTest([5, 6]);
        var inputs = tfc.zeros([4, 3, 2]);
        var rnn = tfl.layers.rnn({ cell: cell });
        var initialStates = rnn.getInitialState(inputs);
        expect(initialStates.length).toEqual(2);
        expectTensorsClose(initialStates[0], tfc.zeros([4, 5]));
        expectTensorsClose(initialStates[1], tfc.zeros([4, 6]));
    });
    it('call: 1 state: returnSequences=false, returnState=false', function () {
        var cell = new RNNCellForTest(4);
        var rnn = tfl.layers.rnn({ cell: cell });
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        var outputs = rnn.apply(inputs);
        expectTensorsClose(outputs, K.scalarTimesArray(scalar(-57.75), tfc.ones([2, 4])));
    });
    it('apply: 1 state: returnSequences=true, returnState=false', function () {
        var cell = new RNNCellForTest(3);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true });
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        var outputs = rnn.apply(inputs);
        expectTensorsClose(outputs, tensor3d([
            [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
            ],
            [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
            ],
        ], [2, 3, 3]));
    });
    it('apply: 1 state: returnSequences=true, returnState=true', function () {
        var cell = new RNNCellForTest(3);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true, returnState: true });
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        var outputs = rnn.apply(inputs);
        expect(outputs.length).toEqual(2);
        expectTensorsClose(outputs[0], tensor3d([
            [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
            ],
            [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
            ],
        ], [2, 3, 3]));
        expectTensorsClose(outputs[1], tensor2d([[57.75, 57.75, 57.75], [57.75, 57.75, 57.75]], [2, 3]));
    });
    it('apply: 2 states: returnSequences=true, returnState=true', function () {
        var cell = new RNNCellForTest([3, 4]);
        var rnn = tfl.layers.rnn({ cell: cell, returnSequences: true, returnState: true });
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        var outputs = rnn.apply(inputs);
        expect(outputs.length).toEqual(3);
        expectTensorsClose(outputs[0], tensor3d([
            [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
            ],
            [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
            ],
        ], [2, 3, 3]));
        expectTensorsClose(outputs[1], tensor2d([[57.75, 57.75, 57.75], [57.75, 57.75, 57.75]], [2, 3]));
        expectTensorsClose(outputs[2], tensor2d([[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]], [2, 4]));
    });
    it('call: with 1 initialState', function () {
        var cell = new RNNCellForTest(4);
        var rnn = tfl.layers.rnn({ cell: cell });
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        var outputs = rnn.apply(inputs, { 'initialState': [tfc.ones([2, 4])] });
        expectTensorsClose(outputs, K.scalarTimesArray(scalar(-58.75), tfc.ones([2, 4])));
    });
    it('call: with 2 initialStates', function () {
        var cell = new RNNCellForTest([4, 5]);
        var rnn = tfl.layers.rnn({ cell: cell, returnState: true });
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        var outputs = rnn.apply(inputs, {
            'initialState': [tfc.ones([2, 4]), K.scalarTimesArray(scalar(2), tfc.ones([2, 5]))]
        });
        expect(outputs.length).toEqual(3);
        expectTensorsClose(outputs[0], K.scalarTimesArray(scalar(-58.75), tfc.ones([2, 4])));
        expectTensorsClose(outputs[1], K.scalarTimesArray(scalar(58.75), tfc.ones([2, 4])));
        expectTensorsClose(outputs[2], K.scalarTimesArray(scalar(59.75), tfc.ones([2, 5])));
    });
    it('call with incorrect number of initialStates leads to ValueError', function () {
        var cell = new RNNCellForTest([4, 5]);
        var rnn = tfl.layers.rnn({ cell: cell, returnState: true });
        var inputs = tensor3d([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
        expect(function () { return rnn.apply(inputs, {
            'initialState': [tfc.ones([2, 4])]
        }); }).toThrowError(/An initialState was passed that is not compatible with/);
    });
});
describeMathCPU('SimpleRNN Symbolic', function () {
    it('returnSequences=false, returnState=false', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var simpleRNN = tfl.layers.simpleRNN({ units: 5 });
        var output = simpleRNN.apply(input);
        expect(output.shape).toEqual([9, 5]);
    });
    it('returnSequences=false, returnState=true', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var simpleRNN = tfl.layers.simpleRNN({ units: 5, returnState: true });
        var output = simpleRNN.apply(input);
        expect(output.length).toEqual(2);
        expect(output[0].shape).toEqual([9, 5]);
        expect(output[1].shape).toEqual([9, 5]);
    });
    it('returnSequences=true, returnState=false', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var simpleRNN = tfl.layers.simpleRNN({ units: 5, returnSequences: true });
        var output = simpleRNN.apply(input);
        expect(output.shape).toEqual([9, 10, 5]);
    });
    it('returnSequences=true, returnState=true', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var simpleRNN = tfl.layers.simpleRNN({
            units: 5,
            returnSequences: true,
            returnState: true,
        });
        var output = simpleRNN.apply(input);
        expect(output.length).toEqual(2);
        expect(output[0].shape).toEqual([9, 10, 5]);
        expect(output[1].shape).toEqual([9, 5]);
    });
    it('Serialization round trip', function () {
        var layer = tfl.layers.simpleRNN({ units: 4 });
        var pythonicConfig = convertTsToPythonic(layer.getConfig());
        var tsConfig = convertPythonicToTs(pythonicConfig);
        var layerPrime = tfl.layers.simpleRNN(tsConfig);
        expect(layerPrime.getConfig().units).toEqual(4);
    });
});
describeMathCPUAndGPU('SimpleRNN Tensor', function () {
    var units = 5;
    var batchSize = 4;
    var inputSize = 2;
    var activations = ['linear', 'tanh'];
    var _loop_1 = function (activation) {
        var testTitle = "returnSequences=false, returnState=false, useBias=true, " + activation;
        it(testTitle, function () {
            var timeSteps = 1;
            var simpleRNN = tfl.layers.simpleRNN({
                units: units,
                kernelInitializer: 'ones',
                recurrentInitializer: 'ones',
                biasInitializer: 'ones',
                activation: activation
            });
            var input = tfc.ones([batchSize, timeSteps, inputSize]);
            var output = simpleRNN.apply(input);
            var expectedElementValue = inputSize + 1;
            if (activation === 'tanh') {
                expectedElementValue = Math.tanh(expectedElementValue);
            }
            expectTensorsClose(output, K.scalarTimesArray(scalar(expectedElementValue), tfc.ones([batchSize, units])));
        });
    };
    for (var _i = 0, activations_1 = activations; _i < activations_1.length; _i++) {
        var activation = activations_1[_i];
        _loop_1(activation);
    }
    var returnStateValues = [false, true];
    var _loop_2 = function (returnState) {
        var testTitle = "returnSequences=true, " +
            ("returnState=" + returnState + ", useBias=true, linear");
        it(testTitle, function () {
            var timeSteps = 2;
            var simpleRNN = tfl.layers.simpleRNN({
                units: units,
                returnSequences: true,
                returnState: returnState,
                kernelInitializer: 'ones',
                recurrentInitializer: 'ones',
                biasInitializer: 'ones',
                activation: 'linear'
            });
            var input = tfc.ones([batchSize, timeSteps, inputSize]);
            var output = simpleRNN.apply(input);
            var finalState;
            if (returnState) {
                output = output;
                expect(output.length).toEqual(2);
                finalState = output[1];
                output = output[0];
            }
            else {
                output = output;
            }
            expect(output.shape).toEqual([batchSize, timeSteps, units]);
            var timeMajorOutput = tfc.transpose(output, [1, 0, 2]);
            var outputT0 = K.sliceAlongFirstAxis(timeMajorOutput, 0, 1);
            var outputT1 = K.sliceAlongFirstAxis(timeMajorOutput, 1, 1);
            expectTensorsClose(outputT0, K.scalarTimesArray(scalar(inputSize + 1), tfc.ones([1, batchSize, units])));
            expectTensorsClose(outputT1, K.scalarTimesArray(scalar((inputSize + 1) * (units + 1)), tfc.ones([1, batchSize, units])));
            if (returnState) {
                expectTensorsClose(finalState, outputT1.reshape([batchSize, units]));
            }
        });
    };
    for (var _a = 0, returnStateValues_1 = returnStateValues; _a < returnStateValues_1.length; _a++) {
        var returnState = returnStateValues_1[_a];
        _loop_2(returnState);
    }
    it('BPTT', function () {
        var sequenceLength = 3;
        var inputSize = 4;
        var batchSize = 5;
        var simpleRNN = tfl.layers.simpleRNN({
            units: 1,
            kernelInitializer: 'ones',
            recurrentInitializer: 'ones',
            useBias: false,
        });
        var dense = tfl.layers.dense({
            units: 1,
            kernelInitializer: 'ones',
            useBias: false,
        });
        var sgd = tfc.train.sgd(5);
        var x = tfc.ones([batchSize, sequenceLength, inputSize]);
        var y = tfc.zeros([batchSize, 1]);
        dense.apply(simpleRNN.apply(x));
        var lossFn = function () {
            return tfc.mean(metrics.mse(y, dense.apply(simpleRNN.apply(x))))
                .asScalar();
        };
        for (var i = 0; i < 2; ++i) {
            sgd.minimize(lossFn);
        }
        expectTensorsClose(simpleRNN.getWeights()[0], K.scalarTimesArray(scalar(0.8484658), tfc.ones([4, 1])));
        expectTensorsClose(simpleRNN.getWeights()[1], K.scalarTimesArray(scalar(0.8484799), tfc.ones([1, 1])));
        expectTensorsClose(dense.getWeights()[0], K.scalarTimesArray(scalar(80.967026), tfc.ones([1, 1])));
    });
});
describeMathCPU('GRU Symbolic', function () {
    it('returnSequences=false, returnState=false', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var gru = tfl.layers.gru({ units: 5 });
        var output = gru.apply(input);
        expect(output.shape).toEqual([9, 5]);
    });
    it('returnSequences=false, returnState=true', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var gru = tfl.layers.gru({ units: 5, returnState: true });
        var output = gru.apply(input);
        expect(output.length).toEqual(2);
        expect(output[0].shape).toEqual([9, 5]);
        expect(output[1].shape).toEqual([9, 5]);
    });
    it('returnSequences=true, returnState=false', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var gru = tfl.layers.gru({ units: 5, returnSequences: true });
        var output = gru.apply(input);
        expect(output.shape).toEqual([9, 10, 5]);
    });
    it('returnSequences=true, returnState=true', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var gru = tfl.layers.gru({
            units: 5,
            returnSequences: true,
            returnState: true,
        });
        var output = gru.apply(input);
        expect(output.length).toEqual(2);
        expect(output[0].shape).toEqual([9, 10, 5]);
        expect(output[1].shape).toEqual([9, 5]);
    });
    it('trainableWeights, nonTrainableWeights and weights give correct outputs', function () {
        var input = new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
        var gru = tfl.layers.gru({ units: 5, returnState: true });
        gru.apply(input);
        expect(gru.trainable).toEqual(true);
        expect(gru.trainableWeights.length).toEqual(3);
        expect(gru.nonTrainableWeights.length).toEqual(0);
        expect(gru.weights.length).toEqual(3);
    });
    it('Serialization round trip', function () {
        var layer = tfl.layers.gru({ units: 4 });
        var pythonicConfig = convertTsToPythonic(layer.getConfig());
        var tsConfig = convertPythonicToTs(pythonicConfig);
        var layerPrime = tfl.layers.gru(tsConfig);
        expect(layerPrime.getConfig().units).toEqual(4);
    });
});
describeMathCPUAndGPU('GRU Tensor', function () {
    var units = 5;
    var batchSize = 4;
    var inputSize = 2;
    var timeSteps = 3;
    var goldenOutputElementValues = [0.22847827, 0.2813754, 0.29444352];
    var implementations = [1, 2];
    var returnStateValues = [false, true];
    var returnSequencesValues = [false, true];
    var _loop_3 = function (implementation) {
        var _loop_4 = function (returnState) {
            var _loop_5 = function (returnSequences) {
                var testTitle = "implementation=" + implementation + ", " +
                    ("returnSequences=" + returnSequences + ", ") +
                    ("returnState=" + returnState);
                it(testTitle, function () {
                    var gru = tfl.layers.gru({
                        units: units,
                        kernelInitializer: 'ones',
                        recurrentInitializer: 'ones',
                        biasInitializer: 'ones',
                        returnState: returnState,
                        returnSequences: returnSequences,
                        implementation: implementation
                    });
                    var input = tfc.zeros([batchSize, timeSteps, inputSize]);
                    var output = gru.apply(input);
                    var goldenOutputElementValueFinal = goldenOutputElementValues[goldenOutputElementValues.length - 1];
                    var expectedOutput;
                    if (returnSequences) {
                        var outputs = goldenOutputElementValues.map(function (value) { return K.scalarTimesArray(scalar(value), tfc.ones([1, batchSize, units])); });
                        expectedOutput = tfc.transpose(K.concatAlongFirstAxis(K.concatAlongFirstAxis(outputs[0], outputs[1]), outputs[2]), [1, 0, 2]);
                    }
                    else {
                        expectedOutput = K.scalarTimesArray(scalar(goldenOutputElementValueFinal), tfc.ones([batchSize, units]));
                    }
                    if (returnState) {
                        output = output;
                        expect(output.length).toEqual(2);
                        expectTensorsClose(output[0], expectedOutput);
                        expectTensorsClose(output[1], K.scalarTimesArray(scalar(goldenOutputElementValueFinal), tfc.ones([batchSize, units])));
                    }
                    else {
                        output = output;
                        expectTensorsClose(output, expectedOutput);
                    }
                });
            };
            for (var _i = 0, returnSequencesValues_1 = returnSequencesValues; _i < returnSequencesValues_1.length; _i++) {
                var returnSequences = returnSequencesValues_1[_i];
                _loop_5(returnSequences);
            }
        };
        for (var _i = 0, returnStateValues_2 = returnStateValues; _i < returnStateValues_2.length; _i++) {
            var returnState = returnStateValues_2[_i];
            _loop_4(returnState);
        }
    };
    for (var _i = 0, implementations_1 = implementations; _i < implementations_1.length; _i++) {
        var implementation = implementations_1[_i];
        _loop_3(implementation);
    }
    it('BPTT', function () {
        var sequenceLength = 3;
        var inputSize = 4;
        var batchSize = 5;
        var gru = tfl.layers.gru({
            units: 1,
            kernelInitializer: 'zeros',
            recurrentInitializer: 'zeros',
            useBias: false
        });
        var dense = tfl.layers.dense({ units: 1, kernelInitializer: 'ones', useBias: false });
        var sgd = tfc.train.sgd(1);
        var x = tfc.ones([batchSize, sequenceLength, inputSize]);
        var y = tfc.ones([batchSize, 1]);
        dense.apply(gru.apply(x));
        var lossFn = function () {
            return tfc.mean(metrics.mse(y, dense.apply(gru.apply(x))))
                .asScalar();
        };
        for (var i = 0; i < 2; ++i) {
            sgd.minimize(lossFn);
        }
        expectTensorsClose(gru.getWeights()[0], K.tile(tensor2d([[-0.03750037, 0, 1.7500007]], [1, 3]), [4, 1]));
        expectTensorsClose(gru.getWeights()[1], tensor2d([[-1.562513e-02, 0, 2.086183e-07]], [1, 3]));
        expectTensorsClose(dense.getWeights()[0], tensor2d([[1.2187521]], [1, 1]));
    });
});
describeMathCPU('LSTM Symbolic', function () {
    it('returnSequences=false, returnState=false', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var lstm = tfl.layers.lstm({ units: 5 });
        var output = lstm.apply(input);
        expect(output.shape).toEqual([9, 5]);
    });
    it('returnSequences=false, returnState=true', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var lstm = tfl.layers.lstm({ units: 5, returnState: true });
        var output = lstm.apply(input);
        expect(output.length).toEqual(3);
        expect(output[0].shape).toEqual([9, 5]);
        expect(output[1].shape).toEqual([9, 5]);
        expect(output[2].shape).toEqual([9, 5]);
    });
    it('returnSequences=true, returnState=false', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var lstm = tfl.layers.lstm({ units: 5, returnSequences: true });
        var output = lstm.apply(input);
        expect(output.shape).toEqual([9, 10, 5]);
    });
    it('returnSequences=true, returnState=true', function () {
        var input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
        var lstm = tfl.layers.lstm({
            units: 5,
            returnSequences: true,
            returnState: true,
        });
        var output = lstm.apply(input);
        expect(output.length).toEqual(3);
        expect(output[0].shape).toEqual([9, 10, 5]);
        expect(output[1].shape).toEqual([9, 5]);
        expect(output[2].shape).toEqual([9, 5]);
    });
    it('trainableWeights, nonTrainableWeights and weights give correct outputs', function () {
        var input = new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
        var lstm = tfl.layers.lstm({ units: 5, returnState: true });
        lstm.apply(input);
        expect(lstm.trainable).toEqual(true);
        expect(lstm.trainableWeights.length).toEqual(3);
        expect(lstm.nonTrainableWeights.length).toEqual(0);
        expect(lstm.weights.length).toEqual(3);
    });
    it('Serialization round trip', function () {
        var layer = tfl.layers.lstm({ units: 4 });
        var pythonicConfig = convertTsToPythonic(layer.getConfig());
        var tsConfig = convertPythonicToTs(pythonicConfig);
        var layerPrime = tfl.layers.lstm(tsConfig);
        expect(layerPrime.getConfig().units).toEqual(4);
    });
});
describeMathCPUAndGPU('LSTM Tensor', function () {
    var units = 5;
    var batchSize = 4;
    var inputSize = 2;
    var timeSteps = 2;
    var implementations = [1, 2];
    var returnStateValues = [false, true];
    var returnSequencesValues = [false, true];
    var _loop_6 = function (implementation) {
        var _loop_7 = function (returnState) {
            var _loop_8 = function (returnSequences) {
                var testTitle = "implementation=" + implementation + ", " +
                    ("returnSequences=" + returnSequences + ", ") +
                    ("returnState=" + returnState);
                it(testTitle, function () {
                    var lstm = tfl.layers.lstm({
                        units: units,
                        kernelInitializer: 'ones',
                        recurrentInitializer: 'ones',
                        biasInitializer: 'ones',
                        returnState: returnState,
                        returnSequences: returnSequences,
                        implementation: implementation
                    });
                    var input = tfc.ones([batchSize, timeSteps, inputSize]);
                    var output = lstm.apply(input);
                    var goldenOutputElementValueAtT0 = 0.7595095;
                    var goldenOutputElementValueAtT1 = 0.96367633;
                    var goldenHStateElementValue = goldenOutputElementValueAtT1;
                    var goldenCStateElementValue = 1.99505234;
                    var expectedOutput;
                    if (returnSequences) {
                        var outputAtT0 = K.scalarTimesArray(scalar(goldenOutputElementValueAtT0), tfc.ones([1, batchSize, units]));
                        var outputAtT1 = K.scalarTimesArray(scalar(goldenOutputElementValueAtT1), tfc.ones([1, batchSize, units]));
                        expectedOutput = tfc.transpose(K.concatAlongFirstAxis(outputAtT0, outputAtT1), [1, 0, 2]);
                    }
                    else {
                        expectedOutput = K.scalarTimesArray(scalar(goldenOutputElementValueAtT1), tfc.ones([batchSize, units]));
                    }
                    if (returnState) {
                        output = output;
                        expect(output.length).toEqual(3);
                        expectTensorsClose(output[0], expectedOutput);
                        expectTensorsClose(output[1], K.scalarTimesArray(scalar(goldenHStateElementValue), tfc.ones([batchSize, units])));
                        expectTensorsClose(output[2], K.scalarTimesArray(scalar(goldenCStateElementValue), tfc.ones([batchSize, units])));
                    }
                    else {
                        output = output;
                        expectTensorsClose(output, expectedOutput);
                    }
                });
            };
            for (var _i = 0, returnSequencesValues_2 = returnSequencesValues; _i < returnSequencesValues_2.length; _i++) {
                var returnSequences = returnSequencesValues_2[_i];
                _loop_8(returnSequences);
            }
        };
        for (var _i = 0, returnStateValues_3 = returnStateValues; _i < returnStateValues_3.length; _i++) {
            var returnState = returnStateValues_3[_i];
            _loop_7(returnState);
        }
        it('BPTT', function () {
            var sequenceLength = 3;
            var inputSize = 4;
            var batchSize = 5;
            var lstm = tfl.layers.lstm({
                units: 1,
                kernelInitializer: 'zeros',
                recurrentInitializer: 'zeros',
                useBias: false,
            });
            var dense = tfl.layers.dense({
                units: 1,
                kernelInitializer: 'ones',
                useBias: false,
            });
            var sgd = tfc.train.sgd(1);
            var x = tfc.ones([batchSize, sequenceLength, inputSize]);
            var y = tfc.ones([batchSize, 1]);
            dense.apply(lstm.apply(x));
            var lossFn = function () {
                return tfc.mean(metrics.mse(y, dense.apply(lstm.apply(x))))
                    .asScalar();
            };
            for (var i = 0; i < 2; ++i) {
                sgd.minimize(lossFn);
            }
            expectTensorsClose(lstm.getWeights()[0], K.tile(tensor2d([[0.11455188, 0.06545822, 0.8760446, 0.18237013]], [1, 4]), [4, 1]));
            expectTensorsClose(lstm.getWeights()[1], tensor2d([[0.02831176, 0.01934617, 0.00025817, 0.05784169]], [1, 4]));
            expectTensorsClose(dense.getWeights()[0], tensor2d([[1.4559253]], [1, 1]));
        });
    };
    for (var _i = 0, implementations_2 = implementations; _i < implementations_2.length; _i++) {
        var implementation = implementations_2[_i];
        _loop_6(implementation);
    }
});
describeMathCPU('LSTM-deserialization', function () {
    it('modelFromConfig', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            modelFromJSON(fakeLSTMModel)
                .then(function (model) {
                var encoderInputs = tfc.zeros([1, 3, 71], 'float32');
                var decoderInputs = tfc.zeros([1, 3, 94], 'float32');
                var outputs = model.predict([encoderInputs, decoderInputs]);
                expect(outputs.shape).toEqual([1, 3, 94]);
                done();
            })
                .catch(done.fail);
            return [2];
        });
    }); });
});
var fakeLSTMModel = {
    modelTopology: {
        'class_name': 'Model',
        'keras_version': '2.1.2',
        'config': {
            'layers': [
                {
                    'class_name': 'InputLayer',
                    'config': {
                        'dtype': 'float32',
                        'batch_input_shape': [null, null, 71],
                        'name': 'input_1',
                        'sparse': false
                    },
                    'inbound_nodes': [],
                    'name': 'input_1'
                },
                {
                    'class_name': 'InputLayer',
                    'config': {
                        'dtype': 'float32',
                        'batch_input_shape': [null, null, 94],
                        'name': 'input_2',
                        'sparse': false
                    },
                    'inbound_nodes': [],
                    'name': 'input_2'
                },
                {
                    'class_name': 'LSTM',
                    'config': {
                        'recurrent_activation': 'hard_sigmoid',
                        'trainable': true,
                        'recurrent_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'seed': null,
                                'mode': 'fan_avg'
                            }
                        },
                        'use_bias': true,
                        'bias_regularizer': null,
                        'return_state': true,
                        'unroll': false,
                        'activation': 'tanh',
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'units': 256,
                        'unit_forget_bias': true,
                        'activity_regularizer': null,
                        'recurrent_dropout': 0.0,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'seed': null,
                                'mode': 'fan_avg'
                            }
                        },
                        'kernel_constraint': null,
                        'dropout': 0.0,
                        'stateful': false,
                        'recurrent_regularizer': null,
                        'name': 'lstm_1',
                        'bias_constraint': null,
                        'go_backwards': false,
                        'implementation': 1,
                        'kernel_regularizer': null,
                        'return_sequences': false,
                        'recurrent_constraint': null
                    },
                    'inbound_nodes': [[['input_1', 0, 0, {}]]],
                    'name': 'lstm_1'
                },
                {
                    'class_name': 'LSTM',
                    'config': {
                        'recurrent_activation': 'hard_sigmoid',
                        'trainable': true,
                        'recurrent_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'seed': null,
                                'mode': 'fan_avg'
                            }
                        },
                        'use_bias': true,
                        'bias_regularizer': null,
                        'return_state': true,
                        'unroll': false,
                        'activation': 'tanh',
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'units': 256,
                        'unit_forget_bias': true,
                        'activity_regularizer': null,
                        'recurrent_dropout': 0.0,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'seed': null,
                                'mode': 'fan_avg'
                            }
                        },
                        'kernel_constraint': null,
                        'dropout': 0.0,
                        'stateful': false,
                        'recurrent_regularizer': null,
                        'name': 'lstm_2',
                        'bias_constraint': null,
                        'go_backwards': false,
                        'implementation': 1,
                        'kernel_regularizer': null,
                        'return_sequences': true,
                        'recurrent_constraint': null
                    },
                    'inbound_nodes': [[
                            ['input_2', 0, 0, {}], ['lstm_1', 0, 1, {}], ['lstm_1', 0, 2, {}]
                        ]],
                    'name': 'lstm_2'
                },
                {
                    'class_name': 'Dense',
                    'config': {
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'seed': null,
                                'mode': 'fan_avg'
                            }
                        },
                        'name': 'dense_1',
                        'kernel_constraint': null,
                        'bias_regularizer': null,
                        'bias_constraint': null,
                        'activation': 'softmax',
                        'trainable': true,
                        'kernel_regularizer': null,
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'units': 94,
                        'use_bias': true,
                        'activity_regularizer': null
                    },
                    'inbound_nodes': [[['lstm_2', 0, 0, {}]]],
                    'name': 'dense_1'
                }
            ],
            'input_layers': [['input_1', 0, 0], ['input_2', 0, 0]],
            'output_layers': [['dense_1', 0, 0]],
            'name': 'model_1'
        },
        'backend': 'tensorflow'
    }
};
describeMathCPU('StackedRNNCells Symbolic', function () {
    it('With SimpleRNNCell', function () {
        var stackedRNN = tfl.layers.rnn({
            cell: tfl.layers.stackedRNNCells({
                cells: [
                    tfl.layers.simpleRNNCell({ units: 3, recurrentInitializer: 'glorotNormal' }),
                    tfl.layers.simpleRNNCell({ units: 2, recurrentInitializer: 'glorotNormal' })
                ],
            })
        });
        var input = new tfl.SymbolicTensor('float32', [16, 10, 7], null, [], null);
        var output = stackedRNN.apply(input);
        expect(output.shape).toEqual([16, 2]);
        expect(stackedRNN.trainableWeights.length).toEqual(6);
        expect(stackedRNN.nonTrainableWeights.length).toEqual(0);
        expect(stackedRNN.getWeights()[0].shape).toEqual([7, 3]);
        expect(stackedRNN.getWeights()[1].shape).toEqual([3, 3]);
        expect(stackedRNN.getWeights()[2].shape).toEqual([3]);
        expect(stackedRNN.getWeights()[3].shape).toEqual([3, 2]);
        expect(stackedRNN.getWeights()[4].shape).toEqual([2, 2]);
        expect(stackedRNN.getWeights()[5].shape).toEqual([2]);
    });
    it('With LSTMCell', function () {
        var stackedRNN = tfl.layers.rnn({
            cell: tfl.layers.stackedRNNCells({
                cells: [
                    tfl.layers.lstmCell({ units: 3, recurrentInitializer: 'glorotNormal' }),
                    tfl.layers.lstmCell({ units: 2, recurrentInitializer: 'glorotNormal' })
                ],
            })
        });
        var input = new tfl.SymbolicTensor('float32', [16, 10, 7], null, [], null);
        var output = stackedRNN.apply(input);
        expect(output.shape).toEqual([16, 2]);
        expect(stackedRNN.trainableWeights.length).toEqual(6);
        expect(stackedRNN.nonTrainableWeights.length).toEqual(0);
        expect(stackedRNN.getWeights()[0].shape).toEqual([7, 12]);
        expect(stackedRNN.getWeights()[1].shape).toEqual([3, 12]);
        expect(stackedRNN.getWeights()[2].shape).toEqual([12]);
        expect(stackedRNN.getWeights()[3].shape).toEqual([3, 8]);
        expect(stackedRNN.getWeights()[4].shape).toEqual([2, 8]);
        expect(stackedRNN.getWeights()[5].shape).toEqual([8]);
    });
    it('RNN with cell array creates StackedRNNCell', function () {
        var stackedRNN = tfl.layers.rnn({
            cell: [
                tfl.layers.gruCell({ units: 3, recurrentInitializer: 'glorotNormal' }),
                tfl.layers.gruCell({ units: 2, recurrentInitializer: 'glorotNormal' }),
            ],
        });
        var input = new tfl.SymbolicTensor('float32', [16, 10, 7], null, [], null);
        var output = stackedRNN.apply(input);
        expect(output.shape).toEqual([16, 2]);
        expect(stackedRNN.trainableWeights.length).toEqual(6);
        expect(stackedRNN.nonTrainableWeights.length).toEqual(0);
        expect(stackedRNN.getWeights()[0].shape).toEqual([7, 9]);
        expect(stackedRNN.getWeights()[1].shape).toEqual([3, 9]);
        expect(stackedRNN.getWeights()[2].shape).toEqual([9]);
        expect(stackedRNN.getWeights()[3].shape).toEqual([3, 6]);
        expect(stackedRNN.getWeights()[4].shape).toEqual([2, 6]);
        expect(stackedRNN.getWeights()[5].shape).toEqual([6]);
    });
});
describeMathGPU('StackedRNNCells Tensor', function () {
    it('Forward pass', function () {
        var stackedRNN = tfl.layers.rnn({
            cell: tfl.layers.stackedRNNCells({
                cells: [
                    tfl.layers.simpleRNNCell({
                        units: 3,
                        recurrentInitializer: 'ones',
                        kernelInitializer: 'ones',
                        useBias: false
                    }),
                    tfl.layers.gruCell({
                        units: 2,
                        recurrentInitializer: 'ones',
                        kernelInitializer: 'ones',
                        useBias: false
                    }),
                    tfl.layers.lstmCell({
                        units: 1,
                        recurrentInitializer: 'ones',
                        kernelInitializer: 'ones',
                        useBias: false
                    }),
                ],
            })
        });
        var input = tensor3d([
            [
                [0.1, -0.1, 0.2, -0.2], [-0.1, 0.1, -0.2, 0.2],
                [0.1, 0.1, -0.2, -0.2]
            ],
            [
                [0.05, -0.05, 0.1, -0.1], [-0.05, 0.05, -0.1, 0.1],
                [0.05, 0.05, -0.1, -0.1]
            ]
        ], [2, 3, 4]);
        var output = stackedRNN.apply(input);
        expectTensorsClose(output, tensor2d([[-0.07715216], [-0.05906887]], [2, 1]));
    });
});
//# sourceMappingURL=recurrent_test.js.map