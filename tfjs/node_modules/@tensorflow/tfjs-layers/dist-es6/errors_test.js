import { AttributeError, RuntimeError, ValueError } from './errors';
describe('Error classes', function () {
    var _loop_1 = function (SomeClass) {
        it('pass instanceof tests.', function () {
            var msg = 'Some message';
            var e = new SomeClass(msg);
            expect(e.message).toEqual(msg);
            expect(e instanceof SomeClass).toBe(true);
        });
    };
    for (var _i = 0, _a = [AttributeError, RuntimeError, ValueError]; _i < _a.length; _i++) {
        var SomeClass = _a[_i];
        _loop_1(SomeClass);
    }
});
//# sourceMappingURL=errors_test.js.map