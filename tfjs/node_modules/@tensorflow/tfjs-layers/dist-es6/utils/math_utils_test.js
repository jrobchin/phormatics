import { expectArraysClose } from '@tensorflow/tfjs-core/dist/test_util';
import * as math_utils from './math_utils';
import { describeMathCPU } from './test_utils';
describe('isInteger', function () {
    it('True cases', function () {
        expect(math_utils.isInteger(-103)).toBe(true);
        expect(math_utils.isInteger(0)).toBe(true);
        expect(math_utils.isInteger(1337)).toBe(true);
    });
    it('False cases', function () {
        expect(math_utils.isInteger(-1.03)).toBe(false);
        expect(math_utils.isInteger(0.008)).toBe(false);
        expect(math_utils.isInteger(133.7)).toBe(false);
    });
});
describe('arrayProd', function () {
    it('Full length', function () {
        expect(math_utils.arrayProd([2, 3, 4])).toEqual(24);
        expect(math_utils.arrayProd(new Float32Array([2, 3, 4]))).toEqual(24);
    });
    it('Partial from beginning', function () {
        expect(math_utils.arrayProd([2, 3, 4], null, 2)).toEqual(6);
        expect(math_utils.arrayProd([2, 3, 4], 0, 2)).toEqual(6);
    });
    it('Partial to end', function () {
        expect(math_utils.arrayProd([2, 3, 4], 1)).toEqual(12);
        expect(math_utils.arrayProd([2, 3, 4], 1, 3)).toEqual(12);
    });
    it('Partial no beginninng no end', function () {
        expect(math_utils.arrayProd([2, 3, 4, 5], 1, 3)).toEqual(12);
    });
    it('Empty array', function () {
        expect(math_utils.arrayProd([])).toEqual(1);
    });
});
describeMathCPU('min', function () {
    it('Number array', function () {
        expect(math_utils.min([-100, -200, 150])).toEqual(-200);
    });
    it('Float32Array', function () {
        expect(math_utils.min(new Float32Array([-100, -200, 150]))).toEqual(-200);
    });
});
describeMathCPU('max', function () {
    it('Number array', function () {
        expect(math_utils.max([-100, -200, 150])).toEqual(150);
    });
    it('Float32Array', function () {
        expect(math_utils.max(new Float32Array([-100, -200, 150]))).toEqual(150);
    });
});
describeMathCPU('sum', function () {
    it('Number array', function () {
        expect(math_utils.sum([-100, -200, 150])).toEqual(-150);
    });
    it('Float32Array', function () {
        expect(math_utils.sum(new Float32Array([-100, -200, 150]))).toEqual(-150);
    });
});
describeMathCPU('mean', function () {
    it('Number array', function () {
        expect(math_utils.mean([-100, -200, 150])).toEqual(-50);
    });
    it('Float32Array', function () {
        expect(math_utils.mean(new Float32Array([-100, -200, 150]))).toEqual(-50);
    });
});
describeMathCPU('variance', function () {
    it('Number array', function () {
        expect(math_utils.variance([-100, -200, 150, 50])).toEqual(18125);
    });
    it('Float32Array', function () {
        expect(math_utils.variance(new Float32Array([-100, -200, 150, 50])))
            .toEqual(18125);
    });
});
describeMathCPU('median', function () {
    it('Number array', function () {
        expect(math_utils.median([-100, -200, 150, 50])).toEqual(-25);
    });
    it('Float32Array', function () {
        expect(math_utils.median(new Float32Array([-100, -200, 150, 50])))
            .toEqual(-25);
    });
    it('does not mutate input array', function () {
        var numbers = [-100, -200, 150, 50];
        math_utils.median(numbers);
        expectArraysClose(numbers, [-100, -200, 150, 50]);
    });
});
describe('range', function () {
    it('end > begin', function () {
        expect(math_utils.range(0, 1)).toEqual([0]);
        expect(math_utils.range(0, 5)).toEqual([0, 1, 2, 3, 4]);
        expect(math_utils.range(-10, -5)).toEqual([-10, -9, -8, -7, -6]);
        expect(math_utils.range(-3, 3)).toEqual([-3, -2, -1, 0, 1, 2]);
    });
    it('end === begin', function () {
        expect(math_utils.range(0, 0)).toEqual([]);
        expect(math_utils.range(-2, -2)).toEqual([]);
    });
    it('end < begin throws error', function () {
        expect(function () { return math_utils.range(0, -2); }).toThrowError(/.*-2.*0.*forbidden/);
    });
});
//# sourceMappingURL=math_utils_test.js.map