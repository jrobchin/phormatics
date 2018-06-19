import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './logical_executor';
import { createTensorAttr } from './test_helper';
describe('logical', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.scalar(2)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'logical',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: { a: createTensorAttr(0), b: createTensorAttr(1) },
            children: []
        };
    });
    describe('executeOp', function () {
        ['equal', 'notEqual', 'greater', 'greaterEqual', 'less', 'lessEqual',
            'logicalAnd', 'logicalOr']
            .forEach(function (op) {
            it('should call tfc.' + op, function () {
                var spy = spyOn(tfc, op);
                node.op = op;
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0], input2[0]);
            });
        });
        describe('logicalNot', function () {
            it('should call tfc.logicalNot', function () {
                spyOn(tfc, 'logicalNot');
                node.op = 'logicalNot';
                executeOp(node, { input1: input1 }, context);
                expect(tfc.logicalNot).toHaveBeenCalledWith(input1[0]);
            });
        });
        describe('where', function () {
            it('should call tfc.where', function () {
                spyOn(tfc, 'where');
                node.op = 'where';
                node.inputNames = ['input1', 'input2', 'input3'];
                node.params.condition = createTensorAttr(2);
                var input3 = [tfc.scalar(1)];
                executeOp(node, { input1: input1, input2: input2, input3: input3 }, context);
                expect(tfc.where).toHaveBeenCalledWith(input3[0], input1[0], input2[0]);
            });
        });
    });
});
//# sourceMappingURL=logical_executor_test.js.map