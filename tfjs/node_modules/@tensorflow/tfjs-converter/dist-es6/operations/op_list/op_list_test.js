import * as ajv from 'ajv';
import * as schema from '../op_mapper_schema.json';
import * as arithmetic from './arithmetic.json';
import * as basicMath from './basic_math.json';
import * as convolution from './convolution.json';
import * as creation from './creation.json';
import * as graph from './graph.json';
import * as image from './image.json';
import * as logical from './logical.json';
import * as matrices from './matrices.json';
import * as normalization from './normalization.json';
import * as reduction from './reduction.json';
import * as sliceJoin from './slice_join.json';
import * as transformation from './transformation.json';
describe('OpListTest', function () {
    var jsonValidator = new ajv();
    var validator = jsonValidator.compile(schema);
    beforeEach(function () { });
    describe('validate schema', function () {
        var mappersJson = {
            arithmetic: arithmetic,
            basicMath: basicMath,
            convolution: convolution,
            creation: creation,
            logical: logical,
            image: image,
            graph: graph,
            matrices: matrices,
            normalization: normalization,
            reduction: reduction,
            sliceJoin: sliceJoin,
            transformation: transformation
        };
        Object.keys(mappersJson).forEach(function (key) {
            it('should satisfy the schema: ' + key, function () {
                var valid = validator(mappersJson[key]);
                if (!valid)
                    console.log(validator.errors);
                expect(valid).toBeTruthy();
            });
        });
    });
});
//# sourceMappingURL=op_list_test.js.map