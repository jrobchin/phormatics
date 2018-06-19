import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { createDtypeAttr, createNumberAttr, createNumericArrayAttrFromIndex, createTensorAttr } from './test_helper';
import { executeOp } from './transformation_executor';
describe('transformation', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.tensor1d([1, 1])];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'transformation',
            inputNames: ['input1'],
            inputs: [],
            params: { x: createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('cast', function () {
            it('should call tfc.cast', function () {
                spyOn(tfc, 'cast');
                node.op = 'cast';
                node.params.dtype = createDtypeAttr('float32');
                executeOp(node, { input1: input1 }, context);
                expect(tfc.cast).toHaveBeenCalledWith(input1[0], 'float32');
            });
        });
        describe('expandDims', function () {
            it('should call tfc.expandDims', function () {
                spyOn(tfc, 'expandDims');
                node.op = 'expandDims';
                node.params.axis = createNumberAttr(1);
                executeOp(node, { input1: input1 }, context);
                expect(tfc.expandDims).toHaveBeenCalledWith(input1[0], 1);
            });
        });
        describe('pad', function () {
            it('should call tfc.pad', function () {
                spyOn(tfc, 'pad');
                node.op = 'pad';
                node.params.padding = createNumericArrayAttrFromIndex(1);
                node.params.constantValue = createNumberAttr(1);
                node.inputNames = ['input1', 'input3'];
                var input3 = [tfc.tensor2d([1, 1, 2, 2], [2, 2])];
                executeOp(node, { input1: input1, input3: input3 }, context);
                expect(tfc.pad).toHaveBeenCalledWith(input1[0], [[1, 1], [2, 2]], 1);
            });
        });
        describe('reshape', function () {
            it('should call tfc.reshape', function () {
                spyOn(tfc, 'reshape');
                node.op = 'reshape';
                node.params.shape = createNumericArrayAttrFromIndex(1);
                node.inputNames = ['input1', 'input2'];
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.reshape).toHaveBeenCalledWith(input1[0], [1, 1]);
            });
        });
        describe('squeeze', function () {
            it('should call tfc.squeeze', function () {
                spyOn(tfc, 'squeeze');
                node.op = 'squeeze';
                node.params.axis = createNumberAttr(1);
                executeOp(node, { input1: input1 }, context);
                expect(tfc.squeeze).toHaveBeenCalledWith(input1[0], 1);
            });
        });
    });
});
//# sourceMappingURL=transformation_executor_test.js.map