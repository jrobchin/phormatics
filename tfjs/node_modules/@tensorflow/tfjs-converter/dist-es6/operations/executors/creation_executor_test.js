import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './creation_executor';
import { createDtypeAttr, createNumberAttr, createNumberAttrFromIndex, createNumericArrayAttrFromIndex, createTensorAttr } from './test_helper';
describe('creation', function () {
    var node;
    var input1 = [tfc.tensor1d([1, 2, 3])];
    var input2 = [tfc.scalar(1)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'creation',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: { x: createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('fill', function () {
            it('should call tfc.fill', function () {
                spyOn(tfc, 'fill');
                node.op = 'fill';
                node.params['shape'] = createNumericArrayAttrFromIndex(0);
                node.params['value'] = createNumberAttrFromIndex(1);
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.fill).toHaveBeenCalledWith([1, 2, 3], 1);
            });
        });
        describe('linspace', function () {
            it('should call tfc.linspace', function () {
                spyOn(tfc, 'linspace');
                node.op = 'linspace';
                node.params['start'] = createNumberAttrFromIndex(0);
                node.params['stop'] = createNumberAttrFromIndex(1);
                node.params['num'] = createNumberAttrFromIndex(2);
                node.inputNames = ['input', 'input2', 'input3'];
                var input = [tfc.scalar(0)];
                var input3 = [tfc.scalar(2)];
                executeOp(node, { input: input, input2: input2, input3: input3 }, context);
                expect(tfc.linspace).toHaveBeenCalledWith(0, 1, 2);
            });
        });
        describe('oneHot', function () {
            it('should call tfc.oneHot', function () {
                spyOn(tfc, 'oneHot');
                node.op = 'oneHot';
                node.params['indices'] = createNumericArrayAttrFromIndex(0);
                node.params['depth'] = createNumberAttrFromIndex(1);
                node.params['onValue'] = createNumberAttrFromIndex(2);
                node.params['offValue'] = createNumberAttrFromIndex(3);
                node.inputNames = ['input', 'input2', 'input3', 'input4'];
                var input = [tfc.tensor1d([0])];
                var input3 = [tfc.scalar(2)];
                var input4 = [tfc.scalar(3)];
                executeOp(node, { input: input, input2: input2, input3: input3, input4: input4 }, context);
                expect(tfc.oneHot).toHaveBeenCalledWith([0], 1, 2, 3);
            });
        });
        describe('ones', function () {
            it('should call tfc.ones', function () {
                spyOn(tfc, 'ones');
                node.op = 'ones';
                node.params['shape'] = createNumericArrayAttrFromIndex(0);
                node.params['dtype'] = createDtypeAttr('float32');
                executeOp(node, { input1: input1 }, context);
                expect(tfc.ones).toHaveBeenCalledWith([1, 2, 3], 'float32');
            });
        });
        describe('onesLike', function () {
            it('should call tfc.onesLike', function () {
                spyOn(tfc, 'onesLike');
                node.op = 'onesLike';
                executeOp(node, { input1: input1 }, context);
                expect(tfc.onesLike).toHaveBeenCalledWith(input1[0]);
            });
        });
        describe('range', function () {
            it('should call tfc.range', function () {
                spyOn(tfc, 'range');
                node.op = 'range';
                node.params['start'] = createNumberAttrFromIndex(0);
                node.params['stop'] = createNumberAttr(1);
                node.params['step'] = createNumberAttr(2);
                node.params['dtype'] = createDtypeAttr('float32');
                node.inputNames = ['input', 'input2', 'input3'];
                var input = [tfc.scalar(0)];
                var input3 = [tfc.scalar(2)];
                executeOp(node, { input: input, input2: input2, input3: input3 }, context);
                expect(tfc.range).toHaveBeenCalledWith(0, 1, 2, 'float32');
            });
        });
        describe('randomUniform', function () {
            it('should call tfc.randomUniform', function () {
                spyOn(tfc, 'randomUniform');
                node.op = 'randomUniform';
                node.params['shape'] = createNumericArrayAttrFromIndex(0);
                node.inputNames = ['input1'];
                node.params['maxval'] = createNumberAttr(1);
                node.params['minval'] = createNumberAttr(0);
                node.params['dtype'] = createDtypeAttr('float32');
                node.params['seed'] = createNumberAttr(0);
                executeOp(node, { input1: input1 }, context);
                expect(tfc.randomUniform)
                    .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32');
            });
        });
        describe('truncatedNormal', function () {
            it('should call tfc.truncatedNormal', function () {
                spyOn(tfc, 'truncatedNormal');
                node.op = 'truncatedNormal';
                node.params['shape'] = createNumericArrayAttrFromIndex(0);
                node.inputNames = ['input1'];
                node.params['stdDev'] = createNumberAttr(1);
                node.params['mean'] = createNumberAttr(0);
                node.params['dtype'] = createDtypeAttr('float32');
                node.params['seed'] = createNumberAttr(0);
                executeOp(node, { input1: input1 }, context);
                expect(tfc.truncatedNormal)
                    .toHaveBeenCalledWith([1, 2, 3], 0, 1, 'float32', 0);
            });
        });
        describe('zeros', function () {
            it('should call tfc.zeros', function () {
                spyOn(tfc, 'zeros');
                node.op = 'zeros';
                node.params['shape'] = createNumericArrayAttrFromIndex(0);
                node.params['dtype'] = createDtypeAttr('float32');
                executeOp(node, { input1: input1 }, context);
                expect(tfc.zeros).toHaveBeenCalledWith([1, 2, 3], 'float32');
            });
        });
        describe('zerosLike', function () {
            it('should call tfc.zerosLike', function () {
                spyOn(tfc, 'zerosLike');
                node.op = 'zerosLike';
                executeOp(node, { input1: input1 }, context);
                expect(tfc.zerosLike).toHaveBeenCalledWith(input1[0]);
            });
        });
    });
});
//# sourceMappingURL=creation_executor_test.js.map