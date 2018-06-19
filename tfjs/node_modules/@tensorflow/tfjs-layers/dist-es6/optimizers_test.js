import { AdagradOptimizer, AdamOptimizer, RMSPropOptimizer, SGDOptimizer } from '@tensorflow/tfjs-core';
import { getOptimizer } from './optimizers';
import { describeMathCPU } from './utils/test_utils';
describeMathCPU('getOptimizer', function () {
    it("can instantiate SGD", function () {
        var optimizer = getOptimizer('SGD');
        expect(optimizer instanceof SGDOptimizer).toBe(true);
    });
    it("can instantiate sgd", function () {
        var optimizer = getOptimizer('sgd');
        expect(optimizer instanceof SGDOptimizer).toBe(true);
    });
    it("can instantiate Adam", function () {
        var optimizer = getOptimizer('Adam');
        expect(optimizer instanceof AdamOptimizer).toBe(true);
    });
    it("can instantiate adam", function () {
        var optimizer = getOptimizer('adam');
        expect(optimizer instanceof AdamOptimizer).toBe(true);
    });
    it("can instantiate RMSProp", function () {
        var optimizer = getOptimizer('RMSProp');
        expect(optimizer instanceof RMSPropOptimizer).toBe(true);
    });
    it("can instantiate rmsprop", function () {
        var optimizer = getOptimizer('rmsprop');
        expect(optimizer instanceof RMSPropOptimizer).toBe(true);
    });
    it("can instantiate Adagrad", function () {
        var optimizer = getOptimizer('Adagrad');
        expect(optimizer instanceof AdagradOptimizer).toBe(true);
    });
    it("can instantiate adagrad", function () {
        var optimizer = getOptimizer('adagrad');
        expect(optimizer instanceof AdagradOptimizer).toBe(true);
    });
    it('throws for non-existent optimizer', function () {
        expect(function () { return getOptimizer('not an optimizer'); })
            .toThrowError(/Unknown Optimizer/);
    });
});
//# sourceMappingURL=optimizers_test.js.map