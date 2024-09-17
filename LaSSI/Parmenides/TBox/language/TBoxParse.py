import json

from antlr4 import *

from Parmenides.TBox.SentenceMatch import RWVariable
from Parmenides.TBox.language.parmenides_tboxParser import parmenides_tboxParser
from Parmenides.TBox.language.parmenides_tboxVisitor import parmenides_tboxVisitor
from logical_repr.Sentences import Formula

def unfold_string(d):
    if hasattr(d, "text"):
        d = d.text.strip()
    elif hasattr(d, "getText") and callable(getattr(d, "getText", None)):
        d = d.getText().strip()
    else:
        raise ValueError("Unexpected type: "+type(d).__name__+" for: "+str(d))
    if d == "none":
        return None
    return json.loads(d)



class CommonComponents:
    def __init__(self, ontology_query, replacement_pair, operations, where_theta):
        self.operations = operations
        self.replacement_pair = replacement_pair
        self.ontology_query = ontology_query
        self.where_theta = where_theta

class UpdateOperation:
    cc:CommonComponents
    formula:Formula
    def __init__(self, formula, cc:CommonComponents):
        self.cc = cc
        self.formula = formula

class RewriteOperation:
    cc:CommonComponents
    match:Formula
    rewrite:Formula
    def __init__(self, match, rewrite, cc:CommonComponents):
        self.cc = cc
        self.match = match
        self.rewrite = rewrite

class TBoxVisitor(parmenides_tboxVisitor):

    def visit(self, tree):
        if tree is None:
            return None
        else:
            return super().visit(tree)

    def visitSentence_match(self, ctx: parmenides_tboxParser.Sentence_matchContext):
        from Parmenides.TBox.SimpleDataMatch import SentenceMatch
        if ctx is None:
            return None
        relname = unfold_string(ctx.relname)
        n = int(ctx.n.text)
        fields = []
        for x in ctx.field_match():
            fields.append(self.visitField_match(x))
        fields = tuple(fields)
        asname = unfold_string(ctx.as_name)
        parents = int(ctx.parents.text)
        return SentenceMatch(relname, n, fields, asname, parents)

    def visitField_match(self, ctx: parmenides_tboxParser.Field_matchContext):
        from Parmenides.TBox.SimpleDataMatch import FieldMatch
        n = int(ctx.n.text)
        path = tuple([])
        attr = None
        with_value = None
        as_name = None

        if ctx.data_match_path():
            path = self.visit(ctx.data_match_path())
        if ctx.attr is not None:
            attr = unfold_string(ctx.attr)
        if ctx.withval is not None:
            with_value = unfold_string(ctx.withval)
        if ctx.asname is not None:
            as_name = unfold_string(ctx.asname)

        return FieldMatch("MATCH", n, path, attr, with_value, as_name)

    def visitData_match_path(self, ctx: parmenides_tboxParser.Data_match_pathContext):
        if ctx is None:
            return tuple([])
        return tuple([unfold_string(x) for x in ctx.STRING()])

    def common_class_expand(self, ctx):
        if ctx is None:
            return None
        ontology_query = self.visit(ctx.ontology_query())
        replacement_pair = dict([self.visit(x) for x in ctx.replacement_pair()])
        operations = [self.visit(x) for x in ctx.operations()]
        sentence_match = [] if ctx.sentence_match() is None else [self.visit(x) for x in ctx.sentence_match()]
        return CommonComponents(ontology_query, replacement_pair, operations, sentence_match)

    def visitParmenides_tbox(self, ctx: parmenides_tboxParser.Parmenides_tboxContext):
        return [self.visit(x) for x in ctx.rule_()]

    def visitSubstitutions(self, ctx: parmenides_tboxParser.SubstitutionsContext):
        if ctx is None:
            return None
        cc = self.common_class_expand(ctx)
        return UpdateOperation(self.visit(ctx.formula()), cc)

    def visitInvention(self, ctx: parmenides_tboxParser.InventionContext):
        if ctx is None:
            return None
        cc = self.common_class_expand(ctx)
        return RewriteOperation(self.visit(ctx.formula(0)), self.visit(ctx.formula(1)), cc)

    def visitFparen(self, ctx: parmenides_tboxParser.FparenContext):
        return self.visit(ctx.formula())

    def visitRw_variable(self, ctx: parmenides_tboxParser.Rw_variableContext):
        if ctx is None:
            return None
        return RWVariable(unfold_string(ctx.STRING()))

    def visitVariable(self, ctx: parmenides_tboxParser.VariableContext):
        if ctx is None:
            return None
        name = unfold_string(ctx.name)
        type_ = unfold_string(ctx.type_)
        specification = unfold_string(ctx.specification)
        cop = self.visit(ctx.formula())
        from logical_repr.Sentences import FVariable
        return FVariable(name, type_, specification, cop)

    def visitUnary_predicate(self, ctx: parmenides_tboxParser.Unary_predicateContext):
        if ctx is None:
            return None
        rel = unfold_string(ctx.rel)
        src = self.visit(ctx.arg)
        score = float(ctx.score.text)
        properties = frozenset(dict([self.visit(x) for x in ctx.key_values()]).items())
        from logical_repr.Sentences import FBinaryPredicate
        from logical_repr.Sentences import FUnaryPredicate
        return FUnaryPredicate(rel, src, score, properties)
    def visitBinary_predicate(self, ctx: parmenides_tboxParser.Binary_predicateContext):
        if ctx is None:
            return None
        rel = unfold_string(ctx.rel)
        src = self.visit(ctx.src)
        dst = self.visit(ctx.dst)
        score = float(unfold_string(ctx.score))
        properties = frozenset(dict([self.visit(x) for x in ctx.key_values()]).items())
        from logical_repr.Sentences import FBinaryPredicate
        return FBinaryPredicate(rel, src, dst, score, properties)

    def visitAnd(self, ctx: parmenides_tboxParser.AndContext):
        from logical_repr.Sentences import FAnd
        return FAnd(args=tuple([self.visit(x) for x in ctx.formula()]))

    def visitOr(self, ctx: parmenides_tboxParser.OrContext):
        from logical_repr.Sentences import FOr
        return FOr(args=tuple([self.visit(x) for x in ctx.formula()]))

    def visitNot(self, ctx: parmenides_tboxParser.NotContext):
        from logical_repr.Sentences import FNot
        return FNot(self.visit(ctx.formula()))

    def visitRemove(self, ctx: parmenides_tboxParser.RemoveContext):
        from logical_repr.Sentences import RemovePropertiesFromResult
        return RemovePropertiesFromResult(unfold_string(ctx.STRING()))

    def visitAdd(self, ctx: parmenides_tboxParser.AddContext):
        from logical_repr.Sentences import AddPropertyFromResult
        return AddPropertyFromResult(ofField=unfold_string(ctx.STRING()),toAddInFields=self.visit(ctx.formula()))

    def visitAll_properties(self, ctx: parmenides_tboxParser.All_propertiesContext):
        from logical_repr.Sentences import InheritProperties
        return InheritProperties()

    def visitEdge_match(self, ctx: parmenides_tboxParser.Edge_matchContext):
        return [unfold_string(x) for x in ctx.STRING()]

    def visitIsa_match(self, ctx: parmenides_tboxParser.Isa_matchContext):
        return [unfold_string(x) for x in ctx.STRING()]

    def visitAll_queries(self, ctx: parmenides_tboxParser.All_queriesContext):
        return tuple(["and", [self.visit(x) for x in ctx.ontology_query()]])

    def visitReplacement_pair(self, ctx: parmenides_tboxParser.Replacement_pairContext):
        if ctx is None:
            return tuple([None, None])
        else:
            return tuple(["@"+unfold_string(ctx.src), "@"+unfold_string(ctx.dst)])


    def visitKey_values(self, ctx: parmenides_tboxParser.Key_valuesContext):
        if ctx is None:
            return tuple([None, None])
        else:
            return tuple([unfold_string(ctx.STRING()), tuple([self.visit(x) for x in ctx.formula()])])

    def visitValue(self, ctx: parmenides_tboxParser.ValueContext):
        return unfold_string(ctx.STRING())

    def visitNone(self, ctx: parmenides_tboxParser.NoneContext):
        return None

def parse_query(s):
    from Parmenides.TBox.language.parmenides_tboxLexer import parmenides_tboxLexer
    lexer = parmenides_tboxLexer(InputStream(s))
    stream = CommonTokenStream(lexer)
    parser = parmenides_tboxParser(stream)
    v = TBoxVisitor()
    return v.visit(parser.parmenides_tbox())

def load_tbox_rules(filename:str):
    input_text = ""
    with open(filename) as f:
        input_text = f.read()
    return parse_query(input_text)

if __name__ == "__main__":
    obj = load_tbox_rules("../file.txt")
    print(obj)