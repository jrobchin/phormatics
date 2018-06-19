import { checkDataFormat, checkPaddingMode, checkPoolMode, getUniqueTensorName, isValidTensorName, VALID_DATA_FORMAT_VALUES, VALID_PADDING_MODE_VALUES, VALID_POOL_MODE_VALUES } from './common';
describe('checkDataFormat', function () {
    it('Valid values', function () {
        var extendedValues = VALID_DATA_FORMAT_VALUES.concat([undefined, null]);
        for (var _i = 0, extendedValues_1 = extendedValues; _i < extendedValues_1.length; _i++) {
            var validValue = extendedValues_1[_i];
            checkDataFormat(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return checkDataFormat('foo'); }).toThrowError(/foo/);
        try {
            checkDataFormat('bad');
        }
        catch (e) {
            expect(e).toMatch('DataFormat');
            for (var _i = 0, VALID_DATA_FORMAT_VALUES_1 = VALID_DATA_FORMAT_VALUES; _i < VALID_DATA_FORMAT_VALUES_1.length; _i++) {
                var validValue = VALID_DATA_FORMAT_VALUES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
describe('checkPaddingMode', function () {
    it('Valid values', function () {
        var extendedValues = VALID_PADDING_MODE_VALUES.concat([undefined, null]);
        for (var _i = 0, extendedValues_2 = extendedValues; _i < extendedValues_2.length; _i++) {
            var validValue = extendedValues_2[_i];
            checkPaddingMode(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return checkPaddingMode('foo'); }).toThrowError(/foo/);
        try {
            checkPaddingMode('bad');
        }
        catch (e) {
            expect(e).toMatch('PaddingMode');
            for (var _i = 0, VALID_PADDING_MODE_VALUES_1 = VALID_PADDING_MODE_VALUES; _i < VALID_PADDING_MODE_VALUES_1.length; _i++) {
                var validValue = VALID_PADDING_MODE_VALUES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
describe('checkPoolMode', function () {
    it('Valid values', function () {
        var extendedValues = VALID_POOL_MODE_VALUES.concat([undefined, null]);
        for (var _i = 0, extendedValues_3 = extendedValues; _i < extendedValues_3.length; _i++) {
            var validValue = extendedValues_3[_i];
            checkPoolMode(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return checkPoolMode('foo'); }).toThrowError(/foo/);
        try {
            checkPoolMode('bad');
        }
        catch (e) {
            expect(e).toMatch('PoolMode');
            for (var _i = 0, VALID_POOL_MODE_VALUES_1 = VALID_POOL_MODE_VALUES; _i < VALID_POOL_MODE_VALUES_1.length; _i++) {
                var validValue = VALID_POOL_MODE_VALUES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
describe('isValidTensorName', function () {
    it('Valid tensor names', function () {
        expect(isValidTensorName('a')).toEqual(true);
        expect(isValidTensorName('A')).toEqual(true);
        expect(isValidTensorName('foo1')).toEqual(true);
        expect(isValidTensorName('Foo2')).toEqual(true);
        expect(isValidTensorName('n_1')).toEqual(true);
        expect(isValidTensorName('n.1')).toEqual(true);
        expect(isValidTensorName('n_1_2')).toEqual(true);
        expect(isValidTensorName('n.1.2')).toEqual(true);
        expect(isValidTensorName('a/B/c')).toEqual(true);
        expect(isValidTensorName('z_1/z_2/z.3')).toEqual(true);
    });
    it('Invalid tensor names: empty', function () {
        expect(isValidTensorName('')).toEqual(false);
    });
    it('Invalid tensor names: whitespaces', function () {
        expect(isValidTensorName('a b')).toEqual(false);
        expect(isValidTensorName('ab ')).toEqual(false);
    });
    it('Invalid tensor names: forbidden characters', function () {
        expect(isValidTensorName('foo1-2')).toEqual(false);
        expect(isValidTensorName('bar3!4')).toEqual(false);
    });
    it('Invalid tensor names: invalid first characters', function () {
        expect(isValidTensorName('/foo/bar')).toEqual(false);
        expect(isValidTensorName('.baz')).toEqual(false);
        expect(isValidTensorName('_baz')).toEqual(false);
        expect(isValidTensorName('1Qux')).toEqual(false);
    });
    it('Invalid tensor names: non-ASCII', function () {
        expect(isValidTensorName('フ')).toEqual(false);
        expect(isValidTensorName('ξ')).toEqual(false);
    });
});
describe('getUniqueTensorName', function () {
    it('Adds unique suffixes to tensor names', function () {
        expect(getUniqueTensorName('xx')).toEqual('xx');
        expect(getUniqueTensorName('xx')).toEqual('xx_1');
        expect(getUniqueTensorName('xx')).toEqual('xx_2');
        expect(getUniqueTensorName('xx')).toEqual('xx_3');
    });
    it('Correctly handles preexisting unique suffixes on tensor names', function () {
        expect(getUniqueTensorName('yy')).toEqual('yy');
        expect(getUniqueTensorName('yy')).toEqual('yy_1');
        expect(getUniqueTensorName('yy_1')).toEqual('yy_1_1');
        expect(getUniqueTensorName('yy')).toEqual('yy_2');
        expect(getUniqueTensorName('yy_1')).toEqual('yy_1_2');
        expect(getUniqueTensorName('yy_2')).toEqual('yy_2_1');
        expect(getUniqueTensorName('yy')).toEqual('yy_3');
        expect(getUniqueTensorName('yy_1_1')).toEqual('yy_1_1_1');
    });
});
//# sourceMappingURL=common_test.js.map