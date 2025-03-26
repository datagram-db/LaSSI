grammar MatchingLanguage;
// java -jar /home/giacomo/Scaricati/antlr-4.13.2-complete.jar -visitor  -Dlanguage=Python3 MatchingLanguage.g4
language: import_statement* (rule ';')+;

import_statement: IMPORT fun=STRING OVERMODULE module=STRING #function_import
                | IMPORT ALPHANAME  OVERMODULE module=STRING #class_import
                ;

rule: NESTED? MATCHING query=match_multiobj
       (REPLACE (replacement ',')* replacement)?
       (EXTEND (extension ',')* extension )?
       (WHERE prop)?
       AS (rewriting=object|variable|jpath|rewrite_list)
      ;

match_multiobj: (object ',')* object;

prop: LPAR prop RPAR #p_par
    | prop AND prop  #p_and
    | prop OR  prop  #p_or
    | prop IMPL prop #p_impl
    | NOT prop       #p_not
    | prop EQ EMPTY  #p_isempty
    | prop EQ prop   #p_eq
    | prop NEQ prop   #p_neq
    | prop LEQ prop   #p_leq
    | prop LT prop   #p_lt
    | prop GEQ prop   #p_geq
    | prop GT prop   #p_gt
    | prop ISIN prop #p_gt
    | variable       #p_var
    | jpath          #p_jpath
    | PYTHON STRING  #p_json
    | CALL STRING (WITH funarg+)?    #p_call
    ;

extension: fun=STRING (WITH funarg+)?;
object: ALPHANAME LPAR ((object ',')* object)? RPAR                  #actual_object
      | ALPHANAME LPAR ((STRING EQ object ',')* STRING EQ object)? RPAR  #actual_tuple_of_type_and_args
      |  variable                                                                                 #actual_variable
      | STRING '(' object ')' (WITH funarg+)?                         #actual_unary_function_with_args
      |  IGNORE                                                                                   #ignoring_argument
      | '(' object ')'   #par
      | STRING                                                        #actual_string
      ;
jpath : JSONPATH STRING;
variable: 'var' LPAR ALPHANAME RPAR ;
rewrite_list: (SHALLOW|DEEP) REWRITE (rewrite ',')* rewrite;

funarg:      ALPHANAME ':' (variable|(PYTHON? STRING));
replacement: variable           WITH as;
rewrite:     repl   TO as;

as: jpath|object|variable;
repl : variable|jpath;

IMPORT: 'import';
SHALLOW: 'shallow';
DEEP: 'deep';
AS: 'as';
TO: 'to';
VAR: 'var';
MATCHING: 'match';
NESTED: 'nested';
WHERE: 'where';
EXTEND: 'extend-with';
REPLACE: 'replace';
REWRITE: 'rewrite';
WITH: 'with';
OVERMODULE: 'in-module';
CALL: 'call';
AND: '&&';
OR: '||';
LEQ: '⩽'|'≤';
GEQ: '≥'|'⩾';
LT: '<';
GT: '>';
IMPL: '→';
LPAR: '(';
RPAR: ')';
NEQ: '≠';
NOT: '!';
EQ: '=';
ISIN: '∈';
EMPTY: '∅';
JSONPATH: 'jsonpath';
PYTHON: 'python';
ALPHANAME: [a-zA-Z]+;
IGNORE: '_';

STRING : '"' (~[\\"] | '\\' [\\"])* '"';
SPACE : [ \t\r\n]+ -> skip;

COMMENT
    : '/*' .*? '*/' -> skip
;

LINE_COMMENT
    : '//' ~[\r\n]* -> skip
;
