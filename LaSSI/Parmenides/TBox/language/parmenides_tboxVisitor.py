# Generated from Parmenides/TBox/parmenides_tbox.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .parmenides_tboxParser import parmenides_tboxParser
else:
    from parmenides_tboxParser import parmenides_tboxParser

# This class defines a complete generic visitor for a parse tree produced by parmenides_tboxParser.

class parmenides_tboxVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by parmenides_tboxParser#parmenides_tbox.
    def visitParmenides_tbox(self, ctx:parmenides_tboxParser.Parmenides_tboxContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#substitutions.
    def visitSubstitutions(self, ctx:parmenides_tboxParser.SubstitutionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#invention.
    def visitInvention(self, ctx:parmenides_tboxParser.InventionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#fparen.
    def visitFparen(self, ctx:parmenides_tboxParser.FparenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#rw_variable.
    def visitRw_variable(self, ctx:parmenides_tboxParser.Rw_variableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#variable.
    def visitVariable(self, ctx:parmenides_tboxParser.VariableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#unary_predicate.
    def visitUnary_predicate(self, ctx:parmenides_tboxParser.Unary_predicateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#binary_predicate.
    def visitBinary_predicate(self, ctx:parmenides_tboxParser.Binary_predicateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#and.
    def visitAnd(self, ctx:parmenides_tboxParser.AndContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#or.
    def visitOr(self, ctx:parmenides_tboxParser.OrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#not.
    def visitNot(self, ctx:parmenides_tboxParser.NotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#sentence_match.
    def visitSentence_match(self, ctx:parmenides_tboxParser.Sentence_matchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#field_match.
    def visitField_match(self, ctx:parmenides_tboxParser.Field_matchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#data_match_path.
    def visitData_match_path(self, ctx:parmenides_tboxParser.Data_match_pathContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#remove.
    def visitRemove(self, ctx:parmenides_tboxParser.RemoveContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#add.
    def visitAdd(self, ctx:parmenides_tboxParser.AddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#all_properties.
    def visitAll_properties(self, ctx:parmenides_tboxParser.All_propertiesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#edge_match.
    def visitEdge_match(self, ctx:parmenides_tboxParser.Edge_matchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#isa_match.
    def visitIsa_match(self, ctx:parmenides_tboxParser.Isa_matchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#all_queries.
    def visitAll_queries(self, ctx:parmenides_tboxParser.All_queriesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#replacement_pair.
    def visitReplacement_pair(self, ctx:parmenides_tboxParser.Replacement_pairContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#key_values.
    def visitKey_values(self, ctx:parmenides_tboxParser.Key_valuesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#value.
    def visitValue(self, ctx:parmenides_tboxParser.ValueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by parmenides_tboxParser#none.
    def visitNone(self, ctx:parmenides_tboxParser.NoneContext):
        return self.visitChildren(ctx)



del parmenides_tboxParser