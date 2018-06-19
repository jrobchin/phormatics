import { randomUniform, scalar, tensor1d, zeros } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import { nameScope } from './backend/tfjs_backend';
import * as tfl from './index';
import { describeMathCPU, describeMathCPUAndGPU, expectTensorsClose } from './utils/test_utils';
import * as V from './variables';
describeMathCPU('Variable', function () {
    it('Variable constructor: no explicit name', function () {
        var v1 = new V.LayerVariable(zeros([2]));
        expect(v1.name.indexOf('Variable')).toEqual(0);
        expect(v1.dtype).toEqual('float32');
        expect(v1.shape).toEqual([2]);
        expect(v1.trainable).toEqual(true);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));
        var v2 = new V.LayerVariable(zeros([2, 2]));
        expect(v2.name.indexOf('Variable')).toEqual(0);
        expect(v2.dtype).toEqual('float32');
        expect(v2.shape).toEqual([2, 2]);
        expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
        expect(v2.name === v1.name).toBe(false);
    });
    it('Variable constructor: explicit name', function () {
        var v1 = new V.LayerVariable(zeros([]), undefined, 'foo');
        expect(v1.name.indexOf('foo')).toEqual(0);
        expect(v1.dtype).toEqual('float32');
        expect(v1.shape).toEqual([]);
        expect(v1.trainable).toEqual(true);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
        var v2 = new V.LayerVariable(zeros([2, 2, 1]));
        expect(v1.name.indexOf('foo')).toEqual(0);
        expect(v2.dtype).toEqual('float32');
        expect(v2.shape).toEqual([2, 2, 1]);
        expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
        expect(v2.name.length).toBeGreaterThan(0);
        expect(v2.name === v1.name).toBe(false);
    });
    it('Variable constructor: explicit name with name scope', function () {
        var v1;
        nameScope('barScope', function () {
            nameScope('bazScope', function () {
                v1 = new V.LayerVariable(scalar(0), undefined, 'foo');
            });
        });
        expect(v1.name.indexOf('barScope/bazScope/foo')).toEqual(0);
        expect(v1.dtype).toEqual('float32');
        expect(v1.shape).toEqual([]);
        expect(v1.trainable).toEqual(true);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
    });
    it('Variable trainable property', function () {
        var v1 = new V.LayerVariable(zeros([]), null, 'foo', false);
        expect(v1.trainable).toEqual(false);
    });
    it('Variable works if name is null or undefined', function () {
        expect((new V.LayerVariable(zeros([]), null)).name.indexOf('Variable'))
            .toEqual(0);
        expect((new V.LayerVariable(zeros([]), undefined)).name.indexOf('Variable'))
            .toEqual(0);
    });
    it('int32 dtype', function () {
        expect(new V.LayerVariable(zeros([]), 'int32').dtype).toEqual('int32');
    });
    it('bool dtype', function () {
        expect(new V.LayerVariable(zeros([]), 'bool').dtype).toEqual('bool');
    });
    it('Read value', function () {
        var v1 = new V.LayerVariable(scalar(10), null, 'foo');
        expect(v1.read().dataSync()).toEqual(new Float32Array([10]));
    });
    it('Update value: Compatible shape', function () {
        var v = new V.LayerVariable(tensor1d([10, -10]), null, 'bar');
        expect(v.name.indexOf('bar')).toEqual(0);
        expect(v.shape).toEqual([2]);
        expect(v.read().dataSync()).toEqual(new Float32Array([10, -10]));
        v.write(tensor1d([10, 50]));
        expect(v.name.indexOf('bar')).toEqual(0);
        expect(v.shape).toEqual([2]);
        expect(v.read().dataSync()).toEqual(new Float32Array([10, 50]));
    });
    it('Update value: w/ constraint', function () {
        var v = new V.LayerVariable(tensor1d([10, -10]), null, 'bar', true, tfl.constraints.nonNeg());
        v.write(tensor1d([-10, 10]));
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 10]));
    });
    it('Update value: Incompatible shape', function () {
        var v = new V.LayerVariable(zeros([2, 2]), null, 'qux');
        expect(function () {
            v.write(zeros([4]));
        }).toThrowError();
    });
    it('Generates unique ID', function () {
        var v1 = new V.LayerVariable(scalar(1), null, 'foo');
        var v2 = new V.LayerVariable(scalar(1), null, 'foo');
        expect(v1.id).not.toEqual(v2.id);
    });
    it('Generates unique IDs for Tensors and Variables', function () {
        var v1 = scalar(1);
        var v2 = new V.LayerVariable(scalar(1), null, 'foo');
        expect(v1.id).not.toEqual(v2.id);
    });
});
describeMathCPUAndGPU('Create Variable', function () {
    it('From Tensor, no explicit name', function () {
        var v = V.variable(zeros([2, 2]));
        expect(v.name.indexOf('Variable')).toEqual(0);
        expect(v.shape).toEqual([2, 2]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('From Tensor, no explicit name', function () {
        var v = V.variable(zeros([3]));
        expect(v.name.indexOf('Variable')).toEqual(0);
        expect(v.shape).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('From Tensor, explicit name', function () {
        var v = V.variable(zeros([3]), undefined, 'Var1');
        expect(v.name.indexOf('Var1')).toEqual(0);
        expect(v.shape).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
});
describeMathCPUAndGPU('ZerosVariable', function () {
    it('Scalar', function () {
        var s = V.zerosVariable([], 'float32', 'Scalar');
        expect(s.name.indexOf('Scalar')).toEqual(0);
        expect(K.shape(s.read())).toEqual([]);
        expect(s.read().dataSync()).toEqual(new Float32Array([0]));
    });
    it('Vector', function () {
        var v = V.zerosVariable([3], 'float32', 'Vector');
        expect(v.name.indexOf('Vector')).toEqual(0);
        expect(K.shape(v.read())).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('Matrix', function () {
        var m = V.zerosVariable([2, 2], 'float32', 'Matrix');
        expect(m.name.indexOf('Matrix')).toEqual(0);
        expect(K.shape(m.read())).toEqual([2, 2]);
        expect(m.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('3D', function () {
        var t = V.zerosVariable([2, 2, 2], 'float32', 'Tertiary');
        expect(t.name.indexOf('Tertiary')).toEqual(0);
        expect(K.shape(t.read())).toEqual([2, 2, 2]);
        expect(t.read().dataSync()).toEqual(new Float32Array([
            0, 0, 0, 0, 0, 0, 0, 0
        ]));
    });
    it('4D', function () {
        var q = V.zerosVariable([1, 2, 1, 3], 'float32', 'Quaternary');
        expect(q.name.indexOf('Quaternary')).toEqual(0);
        expect(K.shape(q.read())).toEqual([1, 2, 1, 3]);
        expect(q.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
    });
});
describeMathCPUAndGPU('OnesVariable', function () {
    it('Scalar', function () {
        var s = V.onesVariable([], 'float32', 'Scalar');
        expect(s.name.indexOf('Scalar')).toEqual(0);
        expect(K.shape(s.read())).toEqual([]);
        expect(s.read().dataSync()).toEqual(new Float32Array([1]));
    });
    it('Vector', function () {
        var v = V.onesVariable([3], 'float32', 'Vector');
        expect(v.name.indexOf('Vector')).toEqual(0);
        expect(K.shape(v.read())).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([1, 1, 1]));
    });
    it('Matrix', function () {
        var m = V.onesVariable([2, 2], 'float32', 'Matrix');
        expect(m.name.indexOf('Matrix')).toEqual(0);
        expect(K.shape(m.read())).toEqual([2, 2]);
        expect(m.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
    });
    it('3D', function () {
        var t = V.onesVariable([2, 2, 2], 'float32', 'Tertiary');
        expect(t.name.indexOf('Tertiary')).toEqual(0);
        expect(K.shape(t.read())).toEqual([2, 2, 2]);
        expect(t.read().dataSync()).toEqual(new Float32Array([
            1, 1, 1, 1, 1, 1, 1, 1
        ]));
    });
    it('4D', function () {
        var q = V.onesVariable([1, 2, 1, 3], 'float32', 'Quaternary');
        expect(q.name.indexOf('Quaternary')).toEqual(0);
        expect(K.shape(q.read())).toEqual([1, 2, 1, 3]);
        expect(q.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
    });
});
describeMathCPUAndGPU('ZerosLike', function () {
    it('Scalar', function () {
        var s = V.zerosLike(randomUniform([], -10, 10));
        expect(K.shape(s.read())).toEqual([]);
        expect(s.read().dataSync()).toEqual(new Float32Array([0]));
    });
    it('Vector', function () {
        var v = V.zerosLike(randomUniform([3], -10, 10));
        expect(K.shape(v.read())).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('Matrix', function () {
        var m = V.zerosLike(randomUniform([2, 2], -10, 10));
        expect(K.shape(m.read())).toEqual([2, 2]);
        expect(m.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('3D', function () {
        var t = V.zerosLike(randomUniform([2, 2, 2], -10, 10));
        expect(K.shape(t.read())).toEqual([2, 2, 2]);
        expect(t.read().dataSync()).toEqual(new Float32Array([
            0, 0, 0, 0, 0, 0, 0, 0
        ]));
    });
    it('4D', function () {
        var q = V.zerosLike(randomUniform([1, 2, 1, 3], -10, 10));
        expect(K.shape(q.read())).toEqual([1, 2, 1, 3]);
        expect(q.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
    });
});
describeMathCPUAndGPU('OnesLike', function () {
    it('Scalar', function () {
        var s = V.onesLike(randomUniform([], -10, 10));
        expect(K.shape(s.read())).toEqual([]);
        expect(s.read().dataSync()).toEqual(new Float32Array([1]));
    });
    it('Vector', function () {
        var v = V.onesLike(randomUniform([3], -10, 10));
        expect(K.shape(v.read())).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([1, 1, 1]));
    });
    it('Matrix', function () {
        var m = V.onesLike(randomUniform([2, 2], -10, 10));
        expect(K.shape(m.read())).toEqual([2, 2]);
        expect(m.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
    });
    it('3D', function () {
        var t = V.onesLike(randomUniform([2, 2, 2], -10, 10));
        expect(K.shape(t.read())).toEqual([2, 2, 2]);
        expect(t.read().dataSync()).toEqual(new Float32Array([
            1, 1, 1, 1, 1, 1, 1, 1
        ]));
    });
    it('4D', function () {
        var q = V.onesLike(randomUniform([1, 2, 1, 3], -10, 10));
        expect(K.shape(q.read())).toEqual([1, 2, 1, 3]);
        expect(q.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
    });
});
describeMathCPUAndGPU('eye (I-matrix builder)', function () {
    it('Variable Zero sized 2D matrix', function () {
        expect(function () { return V.eyeVariable(0); }).toThrowError(/Shapes can not be <= 0./);
    });
    it('Variable 1 sized 2D matrix', function () {
        var I = V.eyeVariable(1);
        expect(I.shape).toEqual([1, 1]);
        expect(I.read().dataSync()).toEqual(new Float32Array([1]));
    });
    it('Variable 2 sized 2D matrix', function () {
        var I = V.eyeVariable(2);
        expect(I.shape).toEqual([2, 2]);
        expect(I.read().dataSync()).toEqual(new Float32Array([1, 0, 0, 1]));
    });
});
describeMathCPUAndGPU('Variable update', function () {
    it('Update', function () {
        var v = new V.LayerVariable(scalar(10.0));
        V.update(v, scalar(20.0));
        expectTensorsClose(v.read(), scalar(20.0));
    });
    it('Update: Incompatible shape', function () {
        var v = new V.LayerVariable(tensor1d([10.0, 20.0]));
        var x = tensor1d([10.0, 20.0, 30.0]);
        expect(function () { return V.update(v, x); }).toThrowError();
    });
    it('UpdateAdd', function () {
        var v = new V.LayerVariable(scalar(10.0));
        V.updateAdd(v, scalar(20.0));
        expectTensorsClose(v.read(), scalar(30.0));
    });
    it('UpdateAdd: Incompatible shape', function () {
        var v = new V.LayerVariable(tensor1d([10.0, 20.0]));
        var x = tensor1d([0.0, 10.0, 20.0]);
        expect(function () { return V.updateAdd(v, x); }).toThrowError();
    });
    it('UpdateSub', function () {
        var v = new V.LayerVariable(scalar(10.0));
        V.updateSub(v, scalar(20.0));
        var vNew = v.read();
        expectTensorsClose(vNew, scalar(-10.0));
    });
    it('UpdateSub: Incompatible shape', function () {
        var v = new V.LayerVariable(tensor1d([10.0, 20.0]));
        var x = tensor1d([0.0, 10.0, 20.0]);
        expect(function () { return V.updateSub(v, x); }).toThrowError();
    });
});
describeMathCPUAndGPU('batchGetValue', function () {
    it('Legnth-3 Array, Mixed Tensor and Variable', function () {
        var v1 = V.variable(zeros([]));
        var v2 = V.variable(zeros([2]));
        var v3 = V.variable(zeros([2, 2]));
        var values = V.batchGetValue([v1, v2, v3]);
        expect(values.length).toEqual(3);
        expect(values[0].shape).toEqual([]);
        expect(values[0].dataSync()).toEqual(new Float32Array([0]));
        expect(values[1].shape).toEqual([2]);
        expect(values[1].dataSync()).toEqual(new Float32Array([0, 0]));
        expect(values[2].shape).toEqual([2, 2]);
        expect(values[2].dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
});
describeMathCPUAndGPU('batchSetValue', function () {
    it('Update using Tensor values', function () {
        var v1 = V.randomUniformVariable([2], 0, 1);
        var v2 = V.randomUniformVariable([2, 2], 0, 1);
        V.batchSetValue([[v1, zeros([2])], [v2, zeros([2, 2])]]);
        expect(v1.shape).toEqual([2]);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));
        expect(v2.shape).toEqual([2, 2]);
        expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('Update using Tensor values', function () {
        var v1 = V.randomUniformVariable([], 0, 1);
        var v2 = V.randomUniformVariable([2, 2, 1], 0, 1);
        V.batchSetValue([[v1, zeros([])], [v2, zeros([2, 2, 1])]]);
        expect(v1.shape).toEqual([]);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
        expect(v2.shape).toEqual([2, 2, 1]);
        expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('Update empty Array', function () {
        V.batchSetValue([]);
    });
});
//# sourceMappingURL=variables_test.js.map