import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './image_executor';
import { createBoolAttr, createNumericArrayAttr, createTensorAttr } from './test_helper';
describe('image', function () {
    var node;
    var input1 = [tfc.tensor1d([1])];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'input1',
            op: '',
            category: 'image',
            inputNames: ['input1'],
            inputs: [],
            params: {},
            children: []
        };
    });
    describe('executeOp', function () {
        describe('resizeBilinear', function () {
            it('should return input', function () {
                node.op = 'resizeBilinear';
                node.params['images'] = createTensorAttr(0);
                node.params['size'] = createNumericArrayAttr([1, 2]);
                node.params['alignCorners'] = createBoolAttr(true);
                spyOn(tfc.image, 'resizeBilinear');
                executeOp(node, { input1: input1 }, context);
                expect(tfc.image.resizeBilinear)
                    .toHaveBeenCalledWith(input1[0], [1, 2], true);
            });
        });
        describe('resizeNearestNeighbor', function () {
            it('should return input', function () {
                node.op = 'resizeNearestNeighbor';
                node.params['images'] = createTensorAttr(0);
                node.params['size'] = createNumericArrayAttr([1, 2]);
                node.params['alignCorners'] = createBoolAttr(true);
                spyOn(tfc.image, 'resizeNearestNeighbor');
                executeOp(node, { input1: input1 }, context);
                expect(tfc.image.resizeNearestNeighbor)
                    .toHaveBeenCalledWith(input1[0], [1, 2], true);
            });
        });
    });
});
//# sourceMappingURL=image_executor_test.js.map