import { ExecutionContext } from '../executor/execution_context';
import * as arithmetic from './executors/arithmetic_executor';
import * as basic_math from './executors/basic_math_executor';
import * as convolution from './executors/convolution_executor';
import * as creation from './executors/creation_executor';
import * as graph from './executors/graph_executor';
import * as image from './executors/image_executor';
import * as logical from './executors/logical_executor';
import * as matrices from './executors/matrices_executor';
import * as normalization from './executors/normalization_executor';
import * as reduction from './executors/reduction_executor';
import * as slice_join from './executors/slice_join_executor';
import * as transformation from './executors/transformation_executor';
import { executeOp } from './operation_executor';
describe('OperationExecutor', function () {
    var node;
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: 'const',
            category: 'graph',
            inputNames: [],
            inputs: [],
            params: {},
            children: []
        };
    });
    describe('executeOp', function () {
        [arithmetic, basic_math, convolution, creation, image, graph, logical,
            matrices, normalization, reduction, slice_join, transformation]
            .forEach(function (category) {
            it('should call ' + category.CATEGORY + ' executor', function () {
                spyOn(category, 'executeOp');
                node.category = category.CATEGORY;
                executeOp(node, {}, context);
                expect(category.executeOp).toHaveBeenCalledWith(node, {}, context);
            });
        });
    });
});
//# sourceMappingURL=operation_executor_test.js.map