import { Ones, Zeros } from '../initializers';
import { deserialize } from './serialization';
describe('Deserialization', function () {
    it('Zeros Initialzer', function () {
        var config = { className: 'Zeros', config: {} };
        var initializer = deserialize(config);
        expect(initializer instanceof (Zeros)).toEqual(true);
    });
    it('Ones Initialzer', function () {
        var config = { className: 'Ones', config: {} };
        var initializer = deserialize(config);
        expect(initializer instanceof (Ones)).toEqual(true);
    });
});
//# sourceMappingURL=serialization_test.js.map