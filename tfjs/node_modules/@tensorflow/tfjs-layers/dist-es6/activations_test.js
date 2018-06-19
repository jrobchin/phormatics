import { scalar, tensor1d, tensor2d, tensor3d } from '@tensorflow/tfjs-core';
import { Elu, HardSigmoid, Linear, Relu, Relu6, Selu, Sigmoid, Softmax, Softplus, Softsign, Tanh } from './activations';
import { describeMathCPUAndGPU, expectNoLeakedTensors, expectTensorsClose } from './utils/test_utils';
describeMathCPUAndGPU('linear activation', function () {
    var initVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
    var expectedVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
    var linear = new Linear().apply;
    it('1D', function () {
        var initX = tensor1d(initVals);
        expectTensorsClose(linear(initX), tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tensor2d(initVals, [2, 3]);
        expectTensorsClose(linear(initX), tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(linear(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return linear(initX); }, 0);
    });
});
describeMathCPUAndGPU('elu activation', function () {
    var initVals = [-1, 2, 0, 4, -5, 6];
    var expectedVals = initVals.map(function (x) { return x < 0 ? Math.exp(x) - 1 : x; });
    var elu = new Elu().apply;
    it('1D', function () {
        var initX = tensor1d(initVals);
        expectTensorsClose(elu(initX), tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tensor2d(initVals, [2, 3]);
        expectTensorsClose(elu(initX), tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(elu(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return elu(initX); }, 1);
    });
});
describeMathCPUAndGPU('selu activation', function () {
    var initVals = [-1, 2, 0, 4, -5, 6];
    var alpha = 1.6732632423543772848170429916717;
    var scale = 1.0507009873554804934193349852946;
    var expectedVals = initVals.map(function (x) { return scale * (x < 0 ? (alpha * (Math.exp(x) - 1)) : x); });
    var selu = new Selu().apply;
    it('1D', function () {
        var initX = tensor1d(initVals);
        expectTensorsClose(selu(initX), tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tensor2d(initVals, [2, 3]);
        expectTensorsClose(selu(initX), tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(selu(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return selu(initX); }, 1);
    });
});
describeMathCPUAndGPU('relu activation', function () {
    var initVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
    var expectedVals = new Float32Array([0, 2, 0, 4, 0, 6]);
    var relu = new Relu().apply;
    it('1D', function () {
        var initX = tensor1d(initVals);
        expectTensorsClose(relu(initX), tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tensor2d(initVals, [2, 3]);
        expectTensorsClose(relu(initX), tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(relu(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return relu(initX); }, 1);
    });
});
describeMathCPUAndGPU('relu6 activation', function () {
    var initVals = new Float32Array([-10, -5, 0, 1, 5, 15]);
    var expectedVals = new Float32Array([0, 0, 0, 1, 5, 6]);
    var relu6 = new Relu6().apply;
    it('1D', function () {
        var initX = tensor1d(initVals);
        expectTensorsClose(relu6(initX), tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tensor2d(initVals, [2, 3]);
        expectTensorsClose(relu6(initX), tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(relu6(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return relu6(initX); }, 1);
    });
});
describeMathCPUAndGPU('sigmoid activation', function () {
    var sigmoid = new Sigmoid().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    it('Scalar', function () {
        expectTensorsClose(sigmoid(scalar(0)), scalar(0.5));
    });
    it('3D', function () {
        var expectedVals = initVals.map(function (v) { return 1 / (1 + Math.exp(-v)); });
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(sigmoid(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return sigmoid(initX); }, 1);
    });
});
describeMathCPUAndGPU('hardSigmoid activation', function () {
    var hardSigmoid = new HardSigmoid().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    it('Scalar', function () {
        expectTensorsClose(hardSigmoid(scalar(0)), scalar(0.5));
    });
    it('3D', function () {
        var expectedVals = initVals.map(function (v) {
            var y = 0.2 * v + 0.5;
            if (y > 1) {
                return 1;
            }
            else if (y < 0) {
                return 0;
            }
            else {
                return y;
            }
        });
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(hardSigmoid(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return hardSigmoid(initX); }, 1);
    });
});
describeMathCPUAndGPU('softplus activation', function () {
    var softplus = new Softplus().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    it('Scalar', function () {
        expectTensorsClose(softplus(scalar(0)), scalar(Math.log(2)));
    });
    it('3D', function () {
        var expectedVals = initVals.map(function (v) { return Math.log(Math.exp(v) + 1); });
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(softplus(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return softplus(initX); }, 1);
    });
});
describeMathCPUAndGPU('softsign activation', function () {
    var softsign = new Softsign().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    it('Scalar', function () {
        expectTensorsClose(softsign(scalar(0)), scalar(0));
    });
    it('3D', function () {
        var expectedVals = initVals.map(function (v) { return v / (Math.abs(v) + 1); });
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(softsign(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return softsign(initX); }, 1);
    });
});
describeMathCPUAndGPU('tanh activation', function () {
    var tanh = new Tanh().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    var expectedVals = initVals.map(function (x) { return Math.tanh(x); });
    it('1D', function () {
        var initX = tensor1d(initVals);
        expectTensorsClose(tanh(initX), tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tensor2d(initVals, [2, 3]);
        expectTensorsClose(tanh(initX), tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tensor3d(initVals, [1, 2, 3]);
        expectTensorsClose(tanh(initX), tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return tanh(initX); }, 1);
    });
});
describeMathCPUAndGPU('softmax activation', function () {
    var softmax = new Softmax().apply;
    it('1D', function () {
        var initVals = new Float32Array([0, 1, 3, 9]);
        var expectedVals = new Float32Array([0.000, 0.000, 0.002, 0.997]);
        var initX = tensor1d(initVals);
        expectTensorsClose(softmax(initX), tensor1d(expectedVals));
    });
    it('1D all equal', function () {
        var initVals = new Float32Array([-1, -1, -1, -1]);
        var expectedVals = new Float32Array([0.25, 0.25, 0.25, 0.25]);
        var initX = tensor1d(initVals);
        expectTensorsClose(softmax(initX), tensor1d(expectedVals));
    });
    it('2D', function () {
        var initVals = new Float32Array([0, 1, 3, 9, 0, 1, 3, 9]);
        var expectedVals = new Float32Array([0.000, 0.000, 0.002, 0.997, 0.000, 0.000, 0.002, 0.997]);
        var initX = tensor2d(initVals, [2, 4]);
        expectTensorsClose(softmax(initX), tensor2d(expectedVals, [2, 4]));
    });
    it('3D', function () {
        var initVals = new Float32Array([0, 1, 3, 9, 0, 1, 3, 9]);
        var expectedVals = new Float32Array([0.000, 0.000, 0.002, 0.997, 0.000, 0.000, 0.002, 0.997]);
        var initX = tensor3d(initVals, [1, 2, 4]);
        expectTensorsClose(softmax(initX), tensor3d(expectedVals, [1, 2, 4]));
    });
    it('Does not leak', function () {
        var initVals = new Float32Array([0, 1, 3, 9]);
        var initX = tensor1d(initVals);
        expectNoLeakedTensors(function () { return softmax(initX); }, 1);
    });
});
//# sourceMappingURL=activations_test.js.map