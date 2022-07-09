#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <vector>
#include <variant>
#include <stdlib.h>
#include <assert.h>
#include <sys/mman.h>

#define COCK Cock cock { __func__ };
struct Cock {
  char const *name;
  Cock(char const *name) : name(name) { printf("+%s\n", name); }
  ~Cock() { printf("-%s\n", name); }
};

enum {
	Tok_EOF,

	// Punctuation
	Tok_underscore,
	Tok_dot,
	Tok_dot_dot,
	Tok_comma,
	Tok_colon,
	Tok_semicolon,

	Tok_equals,
	Tok_plus,
	Tok_plus_eq,
	Tok_plus_mod,
	Tok_plus_mod_eq,
	Tok_minus,
	Tok_minus_eq,
	Tok_minus_mod,
	Tok_minus_mod_eq,
	Tok_asterisk,
	Tok_multiply_eq,
	Tok_multiply_mod,
	Tok_multiply_mod_eq,
	Tok_divide,
	Tok_divide_eq,
	Tok_mod,
	Tok_mod_eq,
	Tok_open_curly,
	Tok_closing_curly,
	Tok_open_paren,
	Tok_closing_paren,
	Tok_open_square_bracket,
	Tok_closing_square_bracket,

	Tok_double_equals,
	Tok_not_equal,

	Tok_less,
	Tok_less_equal,
	Tok_shift_left,
	Tok_shift_left_eq,

	Tok_greater,
	Tok_greater_equal,
	Tok_shift_right,
	Tok_shift_right_eq,

	Tok_bitand,
	Tok_bitand_eq,
	Tok_bitor,
	Tok_bitor_eq,
	Tok_bitxor,
	Tok_bitxor_eq,

	Tok_logand,
	Tok_logor,
	Tok_lognot,

	Tok_bitnot,

	Tok_identifier,
	Tok_int_literal,
	Tok_char_literal,
	Tok_string_literal,

	Tok_break,
	Tok_case,
	Tok_const,
	Tok_continue,
	Tok_else,
	Tok_endcase,
	Tok_enum,
	Tok_fn,
	Tok_if,
	Tok_loop,
	Tok_return,
	Tok_struct,
	Tok_switch,
	Tok_var,
};

struct Token {
	int kind;
	int offset;
};

struct Tokenizer {
	char const *source;
	int offset;
};

struct Token make_token(int kind, int *offset, int length) {
	*offset += length;
	return (struct Token){kind, *offset - length};
}

constexpr int is_ident_ch(char ch) {
	switch(ch) {
	case 'a'...'z':
	case 'A'...'Z':
	case '0'...'9':
	case '_':
		return 1;
	default:
		return 0;
	}
}

constexpr char hash_base = '0';
constexpr char hash_end = 'z';

constexpr uint64_t keyword_hash(char const *str) {
	uint64_t result = 0;
	do {
		result *= hash_end - hash_base;
		result += *str++ - hash_base;
	} while(is_ident_ch(*str));
	return result;
}

struct Token ident_or_keyword(char const *source, int *offset) {
	uint64_t hash = keyword_hash(source + *offset);

	switch(hash) {
	case keyword_hash("_"): return make_token(Tok_underscore, offset, 1);
	case keyword_hash("break"): return make_token(Tok_break, offset, 5);
	case keyword_hash("case"): return make_token(Tok_case, offset, 4);
	case keyword_hash("const"): return make_token(Tok_const, offset, 5);
	case keyword_hash("continue"): return make_token(Tok_continue, offset, 8);	
	case keyword_hash("else"): return make_token(Tok_else, offset, 4);
	case keyword_hash("endcase"): return make_token(Tok_endcase, offset, 7);
	case keyword_hash("enum"): return make_token(Tok_enum, offset, 4);
	case keyword_hash("fn"): return make_token(Tok_fn, offset, 2);
	case keyword_hash("if"): return make_token(Tok_if, offset, 2);
	case keyword_hash("loop"): return make_token(Tok_loop, offset, 4);
	case keyword_hash("return"): return make_token(Tok_return, offset, 6);
	case keyword_hash("struct"): return make_token(Tok_struct, offset, 6);
	case keyword_hash("switch"): return make_token(Tok_switch, offset, 6);
	case keyword_hash("var"): return make_token(Tok_var, offset, 3);
	}

	int start_offset = *offset;
	do {
		*offset += 1;
	} while(is_ident_ch(source[*offset]));

	return (struct Token){Tok_identifier, start_offset};
}

uint64_t char_value(char ch) {
	if('a' <= ch and ch <= 'z') return ch - 'a' + 0xa;
	if('A' <= ch and ch <= 'Z') return ch - 'A' + 0xA;
	if('0' <= ch and ch <= '9') return ch - '0';
	return ~0;
}

uint64_t int_literal_with_base(char const *source, int *offset, int base, int skip) {
	*offset += skip;
	uint64_t result = 0;
	while(true) {
		if(uint64_t value = char_value(source[*offset]); value < base) {
			*offset += 1;
			result *= base;
			result += value;
		} else {
			return result;
		}
	}
}

uint64_t int_literal_at_offset(char const *source, int *offset) {
	if(source[*offset] == '0') {
		switch(source[*offset + 1]) {
		case 'b': return int_literal_with_base(source, offset, 2, 2);
		case 'x': return int_literal_with_base(source, offset, 16, 2);
		case 'o': return int_literal_with_base(source, offset, 8, 2);
		case 'd': return int_literal_with_base(source, offset, 10, 2);
		default:;
		}
	}
	return int_literal_with_base(source, offset, 10, 0);
}

uint8_t literal_char_value(char const *source, int *offset) {
	if(source[*offset] == '\\') {
		*offset += 1;
		uint8_t value = 0;		
		switch(source[*offset]) {
			case 'x': {
				uint8_t high = char_value(source[*offset + 1]);
				uint8_t low  = char_value(source[*offset + 2]);
				assert(high < 16 && low < 16);
				value = (high << 4) | low;
				offset += 3;
				break;
			}
			case 'n': value = '\n'; *offset += 1; break;
			case 'r': value = '\r'; *offset += 1; break;
			case 't': value = '\t'; *offset += 1; break;
			default:  value = source[*offset]; *offset += 1; break;
		}
		return value;
	}
	*offset += 1;
	return source[*offset - 1];
}

uint64_t char_literal_at_offset(char const *source, int *offset) {
	assert(source[*offset] == '\'');
	*offset += 1;
	int value = literal_char_value(source, offset);
	assert(source[*offset] == '\'');
	*offset += 1;
	return value;
}

void string_literal_at_offset(char const *source, int *offset, uint8_t *output_buf, size_t output_size) {
	assert(source[*offset] == '"');
	*offset += 1;
	while(source[*offset] != '"') {
		uint8_t value = literal_char_value(source, offset);
		if(output_size) {
			*output_buf++ = value;
			--output_size;
		}
	}
	*offset += 1;
	if(output_size) {
		*output_buf = 0;
	}
}

struct Token token_at_offset(char const *source, int *offset) {
	while(true) {
		//printf("Token first char '%c' (0x%02X)\n", source[*offset], source[*offset]);
		switch(source[*offset]) {
		case '\t':
		case '\n':
		case ' ':
			*offset += 1;
			continue;
		case '.':
			if(strncmp(source + *offset, "..", 3) == 0) return make_token(Tok_dot_dot, offset, 3);
			return make_token(Tok_dot, offset, 1);
		case ',': return make_token(Tok_comma, offset, 1);
		case ':': return make_token(Tok_colon, offset, 1);
		case ';': return make_token(Tok_semicolon, offset, 1);
		case '!': if(source[*offset + 1] == '=')
			return make_token(Tok_not_equal, offset, 2);
		else
			return make_token(Tok_lognot, offset, 1);
		case '+':
			switch(source[*offset + 1]) {
			case '=': return make_token(Tok_plus_eq, offset, 2);
			case '%': if(source[*offset + 2] == '=')
				return make_token(Tok_plus_mod_eq, offset, 3);
			else
				return make_token(Tok_plus_mod, offset, 2);
			default: return make_token(Tok_plus, offset, 1);
			}
		case '-':
			switch(source[*offset + 1]) {
			case '=': return make_token(Tok_minus_eq, offset, 2);
			case '%': if(source[*offset + 2] == '=')
				return make_token(Tok_minus_mod_eq, offset, 3);
			else
				return make_token(Tok_minus_mod, offset, 2);
			default: return make_token(Tok_minus, offset, 1);
			}
		case '*':
			switch(source[*offset + 1]) {
			case '=': return make_token(Tok_multiply_eq, offset, 2);
			case '%': if(source[*offset + 2] == '=')
				return make_token(Tok_multiply_mod_eq, offset, 3);
			else
				return make_token(Tok_multiply_mod, offset, 2);
			default: return make_token(Tok_asterisk, offset, 1);
			}
		case '/':
			switch(source[*offset + 1]) {
			case '/': while(source[*offset] != '\n') *offset += 1;
			case '=': return make_token(Tok_divide, offset, 2);
			default: return make_token(Tok_divide_eq, offset, 1);
			}
		case '%': if(source[*offset + 1] == '=')
			return make_token(Tok_mod_eq, offset, 2);
		else
			return make_token(Tok_mod, offset, 1);
		case '<':
			switch(source[*offset + 1]) {
			case '=': return make_token(Tok_less_equal, offset, 2);
			case '<': if(source[*offset + 2] == '=')
				return make_token(Tok_shift_left_eq, offset, 3);
			else
				return make_token(Tok_shift_left, offset, 2);
			default: return make_token(Tok_less, offset, 1);
			}
		case '>':
			switch(source[*offset + 1]) {
			case '=': return make_token(Tok_greater_equal, offset, 2);
			case '>': if(source[*offset + 2] == '=')
				return make_token(Tok_shift_right_eq, offset, 3);
			else
				return make_token(Tok_shift_right, offset, 2);
			default: return make_token(Tok_greater, offset, 1);
			}
		case '&':
			switch(source[*offset + 1]) {
			case '=': return make_token(Tok_bitand_eq, offset, 2);
			case '&': return make_token(Tok_logand, offset, 2);
			default: return make_token(Tok_bitand, offset, 1);
			}
		case '|':
			switch(source[*offset + 1]) {
			case '=': return make_token(Tok_bitor_eq, offset, 2);
			case '|': return make_token(Tok_logor, offset, 2);
			default: return make_token(Tok_bitor, offset, 1);
			}
		case '^': if(source[*offset + 1] == '=')
			return make_token(Tok_bitxor_eq, offset, 2);
		else
			return make_token(Tok_bitxor, offset, 1);
		case '~': return make_token(Tok_bitnot, offset, 1);
		case '{': return make_token(Tok_open_curly, offset, 1);
		case '}': return make_token(Tok_closing_curly, offset, 1);
		case '(': return make_token(Tok_open_paren, offset, 1);
		case ')': return make_token(Tok_closing_paren, offset, 1);
		case '[': return make_token(Tok_open_square_bracket, offset, 1);
		case ']': return make_token(Tok_closing_square_bracket, offset, 1);
		case '=': if(source[*offset + 1] == '=')
			return make_token(Tok_double_equals, offset, 2);
		else
			return make_token(Tok_equals, offset, 1);
		case '\'': {
			int start_offset = *offset;
			char_literal_at_offset(source, offset);
			return (struct Token){Tok_char_literal, start_offset};
		}
		case '"': {
			int start_offset = *offset;
			string_literal_at_offset(source, offset, nullptr, 0);
			return (struct Token){Tok_string_literal, start_offset};	
		}
		case '0'...'9': {
			int start_offset = *offset;
			int_literal_at_offset(source, offset);
			return (struct Token){Tok_int_literal, start_offset};
		}
		case 'a'...'z':
		case 'A'...'Z':
		case '@':
		case '_':
			return ident_or_keyword(source, offset);
		case 0:
			return make_token(Tok_EOF, offset, 0);
		default:
			;
		}
		printf("Unknown character '%c' (0x%02X) in token_at_offset\n", source[*offset], source[*offset]);
		exit(1);
	}
}

struct Token next_token(struct Tokenizer *tok) {
	return token_at_offset(tok->source, &tok->offset);
}

int token_source_length(char const *source, int offset) {
	int result = offset;
	token_at_offset(source, &result);
	return result - offset;
}

auto tokenize(char const *source) {
	struct Tokenizer tok {
		source,
		0,
	};

	std::vector<struct Token> tokens;

	while(1) {
		struct Token token = next_token(&tok);
		if(token.kind == Tok_EOF)
			return tokens;
		tokens.emplace_back(token);
	}
}

struct Declaration {
	int name; // Identifier file offset
	int type; // ASTNode with type
	int value; // ASTNode with value
	bool is_mutable;
};

enum {
	// Offset nodes, has offset into file
	Ast_ident, 
	Ast_int_literal,
	Ast_char_literal,
	Ast_string_literal,

	// Unary expressions, has operand populated
	Ast_unary_plus,
	Ast_unary_minus,
	Ast_unary_bitnot,
	Ast_unary_lognot,
	Ast_pointer_type,
	Ast_addr_of,
	Ast_deref,

	// Binary expressions, has lhs and rhs populated
	Ast_array_type,
	Ast_member_access,
	Ast_array_subscript,
	Ast_multiply,
	Ast_divide,
	Ast_modulus,
	Ast_addition,
	Ast_subtraction,
	Ast_shift_left,
	Ast_shift_right,
	Ast_bitand,
	Ast_bitor,
	Ast_bitxor,
	Ast_less,
	Ast_less_equal,
	Ast_greater,
	Ast_greater_equal,
	Ast_equals,
	Ast_not_equal,
	Ast_logical_and,
	Ast_logical_or,
	Ast_assign,
	Ast_plus_eq,
	Ast_plus_mod,
	Ast_plus_mod_eq,
	Ast_minus_eq,
	Ast_minus_mod,
	Ast_minus_mod_eq,
	Ast_multiply_eq,
	Ast_multiply_mod,
	Ast_multiply_mod_eq,
	Ast_divide_eq,
	Ast_mod_eq,
	Ast_shift_left_eq,
	Ast_shift_right_eq,
	Ast_and_eq,
	Ast_xor_eq,
	Ast_or_eq,

	// Local variable definitions, next is name offset, lhs is type, rhs is initializer
	Ast_local_const,
	Ast_local_var,

	// Function expressions
	Ast_function_expression, // lhs is first parameter, rhs is return type, next is function body
	Ast_function_parameter, // lhs is name offset, rhs is type

	// Function calls
	Ast_function_call, // lhs is callee, next is first argument
	Ast_function_argument, // lhs is value, next is next argument

	// Statements
	Ast_compound_statement,
	Ast_expression_statement,
	Ast_return_statement,

	// Expression or statements
	Ast_if, // next is condition, lhs is taken, rhs is not taken
	Ast_loop, // operand is contents
	Ast_switch, // No one knows, good luck
};

struct ASTNode {
	int kind;
	int next;
	union {
		int offset; // Source offset of identifier
		int operand; // AST node of operand of unary operator
		struct {
			int lhs; // AST node of lhs operand of binary operator
			int rhs; // AST node of rhs operand of binary operator
		};
	};
};

struct AST {
	std::vector<Declaration> decls;
	std::vector<ASTNode> nodes;

	uint64_t eval_comptime_node(char const *source, ASTNode const &node_val) const {
		COCK
		int off = node_val.offset;
		switch(node_val.kind) {
		case Ast_int_literal:  return int_literal_at_offset(source, &off);
		case Ast_char_literal: return char_literal_at_offset(source, &off);
		case Ast_addition:     return eval_comptime_expr(source, node_val.lhs) + eval_comptime_expr(source, node_val.rhs);
		case Ast_subtraction:  return eval_comptime_expr(source, node_val.lhs) - eval_comptime_expr(source, node_val.rhs);
		case Ast_multiply:     return eval_comptime_expr(source, node_val.lhs) * eval_comptime_expr(source, node_val.rhs);
		case Ast_divide:       return eval_comptime_expr(source, node_val.lhs) / eval_comptime_expr(source, node_val.rhs);
		case Ast_modulus:      return eval_comptime_expr(source, node_val.lhs) % eval_comptime_expr(source, node_val.rhs);
		case Ast_bitor:        return eval_comptime_expr(source, node_val.lhs) | eval_comptime_expr(source, node_val.rhs);
		case Ast_bitand:       return eval_comptime_expr(source, node_val.lhs) & eval_comptime_expr(source, node_val.rhs);
		case Ast_bitxor:       return eval_comptime_expr(source, node_val.lhs) ^ eval_comptime_expr(source, node_val.rhs);
		case Ast_shift_left:   return eval_comptime_expr(source, node_val.lhs) << eval_comptime_expr(source, node_val.rhs);
		case Ast_shift_right:  return eval_comptime_expr(source, node_val.lhs) >> eval_comptime_expr(source, node_val.rhs);
		case Ast_ident: {
			auto const hash = keyword_hash(source + off);
			switch(hash) {
			case keyword_hash("undefined"):
			case keyword_hash("false"):
				return 0;
			case keyword_hash("true"):
				return 1;
			}
			for(auto const &decl: decls) {
				auto const decl_hash = keyword_hash(source + decl.name);
				if(hash == decl_hash) {
					return eval_comptime_expr(source, decl.value);
				}
			}
		}
		default:
			assert(false);
		}
	}

	uint64_t eval_comptime_expr(char const *source, int node_idx) const {
		COCK
		assert(node_idx != -1);
		auto const node_val = nodes.at(node_idx);
		return eval_comptime_node(source, node_val);
	}

	uint64_t eval_comptime_type_expr_size(char const *source, int node_idx) const {
		COCK
		assert(node_idx != -1);
		auto const node_val = nodes.at(node_idx);
		switch(node_val.kind) {
		case Ast_ident: {
			auto const hash = keyword_hash(source + node_val.offset);
			switch(hash) {
			case keyword_hash("u0"):
			case keyword_hash("void"):
				return 0;
			case keyword_hash("bool"):
			case keyword_hash("i8"):
			case keyword_hash("u8"):
				return 1;
			case keyword_hash("i16"):
			case keyword_hash("u16"):
				return 2;
			case keyword_hash("i32"):
			case keyword_hash("u32"):
				return 4;
			case keyword_hash("i64"):
			case keyword_hash("u64"):
			case keyword_hash("isize"):
			case keyword_hash("usize"):
				return 8;
			default:
				int namelen = token_source_length(source, node_val.offset);
				printf("Unknown type '%*.s' for size calculation", namelen, source + node_val.offset);
				assert(false);
			}
		}
		case Ast_pointer_type: return 8;
		case Ast_array_type: {
			auto elements = eval_comptime_expr(source, node_val.lhs);
			auto child_size = eval_comptime_type_expr_size(source, node_val.rhs);
			return elements * child_size;
		}
		default:
			printf("Cannot evaluate node kind %d at compile time!", node_val.kind);
			exit(1);
		}
	}
};

void assert_kind(Token const &tok, int kind, char const *message = nullptr) {
	if(tok.kind != kind) {
		printf("Bad token kind %d at offset %d!\n", tok.kind, tok.offset);
		if(message) puts(message);
		exit(1);
	}
}

// Precedence
// 1: Postfix operators, left-to-right
//   Function calls (()), array subscript ([]), member access (.), addrof (.&), deref (.*)
// 2: Prefix operators, right-to-left
//   Unary plus (+), minus (-), logical (!) and bitwise (~) not, array type ([]), pointer type (*)
// 3: Binary operators, left-to-right
//   Multiplication (*), division (/) and remainder (%)
// 4: Binary operators, left-to-right
//   Addition (+), subtraction (-)
// 5: Binary operators, left-to-right
//   Shift operators (<<, >>)
// 6: Binary operators, left-to-right
//   Bitwise operators (&, ^, |)
// 7: Binary operators, left-to-right
//   Relational operators, (<, <=, >, >=, ==, !=)
// 8: Binary operators, left-to-right
//   Logical operators (&&, ||)
// 9: Inplace binary operators, right-to-left
//   (=, +=, +%=, -=, -%=, *=, *%=, /=, %=, <<=, >>=, &=, ^=, |=)

enum {
	Assoc_ltr,
	Assoc_rtl,
};

// Postfix operators are hardcoded to be highest precedence
//#define POSTFIX_PREC 1
#define PREFIX_PREC 2
#define DECLARATION_TYPE_PREC 2
#define NO_PRECEDENCE 99999

struct Parser {
	AST output;

	std::vector<struct Token> const &tokens;
	std::vector<struct Token>::const_iterator it;

	// Leave `it` at the paren token in the param list
	int parse_function_expr() {
		COCK
		assert_kind(*it++, Tok_open_paren);
		std::vector<std::pair<int, int>> fparams;
		while(it->kind != Tok_closing_paren) {
			auto const name = it++;
			assert_kind(*name, Tok_identifier);
			int type = -1;
			if(it->kind == Tok_colon) {
				++it;
				type = parse_expression(DECLARATION_TYPE_PREC);
			}
			fparams.emplace_back(name->offset, type);
			if(it->kind == Tok_comma) {
				++it;
			} else {
				break;
			}
		}
		assert_kind(*it++, Tok_closing_paren);

		int ret_type_expr = parse_expression();
		int block = parse_block();

		int first_param = -1;
		for(auto fparam = fparams.rbegin(); fparam != fparams.rend(); ++fparam) {
			ASTNode n;
			n.kind = Ast_function_parameter;
			n.next = first_param;
			n.lhs = fparam->first;
			n.rhs = fparam->second;
			first_param = add_node(n);
		}
		ASTNode n;
		n.kind = Ast_function_expression;
		n.next = block;
		n.lhs = first_param;
		n.rhs = ret_type_expr;
		return add_node(n);
	}

	int parse_statement() {
		COCK
		int ast_kind;
		switch(it++->kind) {
		case Tok_if: {
			assert_kind(*it++, Tok_open_paren, "Expected '(' after 'if'");
			int condition = parse_expression();
			assert_kind(*it++, Tok_closing_paren, "Expected ')' after if condition");
			int taken = parse_statement();
			int not_taken = -1;
			if(it->kind == Tok_else) {
				++it;
				not_taken = parse_statement();
			}
			ASTNode n;
			n.kind = Ast_if;
			n.next = condition;
			n.lhs = taken;
			n.rhs = not_taken;
			return add_node(n);
		}
		case Tok_const: {
			ast_kind = Ast_local_const;
			if(false) {
			case Tok_var:
				ast_kind = Ast_local_var;
			}

			int name_offset = it++->offset;
			int type = -1;
			if(it->kind == Tok_colon) {
				++it;
				type = parse_expression(DECLARATION_TYPE_PREC);
			}
			assert_kind(*it++, Tok_equals);
			int value = parse_expression();
			assert_kind(*it++, Tok_semicolon);
			ASTNode n;
			n.kind = ast_kind;
			n.next = name_offset;
			n.lhs = type;
			n.rhs = value;
			return add_node(n);
		}
		case Tok_case:
			assert(!"Case label");
		case Tok_loop:
			return add_node({.kind = Ast_loop, .operand = parse_statement()});
		case Tok_break:
			assert(!"Break stmt");
		case Tok_return: {
			if(it->kind == Tok_semicolon) {
				++it;
				return add_node({.kind = Ast_return_statement, .operand = -1});
			}
			int expr = parse_expression();
			assert_kind(*it++, Tok_semicolon);
			return add_node({.kind = Ast_return_statement, .operand = expr});
		}
		case Tok_switch:
			assert(!"Switch stmt");
		case Tok_endcase:
			assert(!"Endcase stmt");
		case Tok_continue:
			assert(!"Continue stmt");
		case Tok_open_curly: --it; return parse_block();
		default: --it;
			int expr = parse_expression();
			assert_kind(*it++, Tok_semicolon);
			return add_node({.kind = Ast_expression_statement, .operand = expr});
		}
	}

	int parse_block() {
		COCK
		std::vector<int> statement_nodes;
		assert_kind(*it++, Tok_open_curly);
		while(it->kind != Tok_closing_curly) {
			statement_nodes.emplace_back(parse_statement());
		}
		int next = -1;
		for(auto stmt = statement_nodes.rbegin(); stmt != statement_nodes.rend(); ++stmt) {
			next = add_node({.kind = Ast_compound_statement, .next = next, .operand = *stmt});
		}
		assert_kind(*it++, Tok_closing_curly);
		return next;
	}

	int add_node(ASTNode node) {
		output.nodes.emplace_back(node);
		return output.nodes.size() - 1;
	}

	int add_bop_node(int kind, int lhs, int rhs) {
		// Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100489
		ASTNode node;
		node.kind = kind;
		node.lhs = lhs;
		node.rhs = rhs;
		return add_node(node);
	}

	int parse_struct_expr() {
		assert(!"Implement struct expr parsing");
	}

	int parse_expression(int precedence = NO_PRECEDENCE) {
		COCK

		// Look for prefix operators or primary expressions
		int lhs = -1;
		switch(it++->kind) {
		// Primary expressions
		case Tok_fn:
			lhs = parse_function_expr();
			break;
		case Tok_identifier:
			lhs = add_node({.kind = Ast_ident, .offset = it[-1].offset});
			break;
		case Tok_int_literal:
			lhs = add_node({.kind = Ast_int_literal, .offset = it[-1].offset});
			break;
		case Tok_char_literal:
			lhs = add_node({.kind = Ast_char_literal, .offset = it[-1].offset});
			break;
		case Tok_string_literal:
			lhs = add_node({.kind = Ast_string_literal, .offset = it[-1].offset});
			break;
		case Tok_dot:
			assert_kind(*it, -1, "TODO: Enum or struct literal expressions");
		case Tok_open_paren:
			lhs = parse_expression();
			assert_kind(*it++, Tok_closing_paren, "Expected ')' after paren expression");
			break;
		case Tok_if:
			assert_kind(*it, -1, "TODO: If expressions");
		case Tok_enum:
			assert_kind(*it, -1, "TODO: Enum expressions");
		case Tok_struct:
			lhs = parse_struct_expr();
			break;
		case Tok_loop:
			assert_kind(*it, -1, "TODO: Loop expressions");
		case Tok_switch:
			assert_kind(*it, -1, "TODO: Switch expressions");

		// Prefix operators
		case Tok_asterisk:
			lhs = add_node({.kind = Ast_pointer_type, parse_expression(PREFIX_PREC)});
			break;
		case Tok_open_square_bracket: {
			int arr_sz = parse_expression();
			assert_kind(*it++, Tok_closing_square_bracket);
			lhs = add_bop_node(Ast_array_type, arr_sz, parse_expression(PREFIX_PREC));
			break;
		}
		case Tok_plus:
			lhs = add_node({.kind = Ast_unary_plus, .operand = parse_expression(PREFIX_PREC)});
			break;
		case Tok_minus:
			lhs = add_node({.kind = Ast_unary_minus, .operand = parse_expression(PREFIX_PREC)});
			break;
		case Tok_bitnot:
			lhs = add_node({.kind = Ast_unary_bitnot, .operand = parse_expression(PREFIX_PREC)});
			break;
		case Tok_lognot:
			lhs = add_node({.kind = Ast_unary_lognot, .operand = parse_expression(PREFIX_PREC)});
			break;
		default:
			assert_kind(*--it, -1, "Expected primary expression!");
			__builtin_unreachable();
			;
		}

		// Look for postfix operators
		do {
			switch(it++->kind) {
			case Tok_open_paren: {
				std::vector<int> fargs;
				while(it->kind != Tok_closing_paren) {
					fargs.emplace_back(parse_expression());
					if(it->kind == Tok_comma) {
						++it;
					} else {
						break;
					}
				}

				int first_farg = -1;
				for(auto farg = fargs.rbegin(); farg != fargs.rend(); ++farg) {
					ASTNode n;
					n.kind = Ast_function_argument;
					n.next = first_farg;
					n.operand = *farg;
					first_farg = add_node(n);
				}
				assert(it++->kind == Tok_closing_paren);
				ASTNode n;
				n.kind = Ast_function_call;
				n.next = first_farg;
				n.operand = lhs;
				lhs = add_node(n);
				continue;
			}
			case Tok_open_square_bracket: {
				int rhs = parse_expression();
				assert_kind(*it++, Tok_closing_square_bracket, "Expected ']' after array subscript");
				lhs = add_bop_node(Ast_array_subscript, lhs, rhs);
				continue;
			}
			case Tok_dot:
				switch(it->kind) {
				case Tok_bitand: ++it;
					lhs = add_node({.kind = Ast_addr_of, .operand = lhs});
					continue;
				case Tok_asterisk: ++it;
					lhs = add_node({.kind = Ast_deref, .operand = lhs});
					continue;
				case Tok_identifier:
					lhs = add_bop_node(Ast_member_access, lhs, it++->offset);
					continue;
				default:
					assert_kind(*it, -1, "Expected identifier, deref or addrof after '.'");
				}
			default: --it; break;
			}
		} while(false);

		do {
			int op = it->kind;
			int ast_kind = -1;
			int op_prec = -1;
			int op_assoc = Assoc_ltr;
			switch(op) {
			break; case Tok_asterisk: op_prec = 3; ast_kind = Ast_multiply;
			break; case Tok_divide:   op_prec = 3; ast_kind = Ast_divide;
			break; case Tok_mod:      op_prec = 3; ast_kind = Ast_modulus;

			break; case Tok_plus:  op_prec = 4; ast_kind = Ast_addition;
			break; case Tok_minus: op_prec = 4; ast_kind = Ast_subtraction;

			break; case Tok_shift_left:  op_prec = 5; ast_kind = Ast_shift_left;
			break; case Tok_shift_right: op_prec = 5; ast_kind = Ast_shift_right;

			break; case Tok_bitand: op_prec = 6; ast_kind = Ast_bitand;
			break; case Tok_bitor:  op_prec = 6; ast_kind = Ast_bitor;
			break; case Tok_bitxor: op_prec = 6; ast_kind = Ast_bitxor;

			break; case Tok_less:          op_prec = 7; ast_kind = Ast_less;
			break; case Tok_less_equal:    op_prec = 7; ast_kind = Ast_less_equal;
			break; case Tok_greater:       op_prec = 7; ast_kind = Ast_greater;
			break; case Tok_greater_equal: op_prec = 7; ast_kind = Ast_greater_equal;
			break; case Tok_double_equals: op_prec = 7; ast_kind = Ast_equals;
			break; case Tok_not_equal:     op_prec = 7; ast_kind = Ast_not_equal;

			break; case Tok_logand: op_prec = 8; ast_kind = Ast_logical_and;
			break; case Tok_logor:  op_prec = 8; ast_kind = Ast_logical_or;

			break; case Tok_equals:          op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_assign;
			break; case Tok_plus_eq:         op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_plus_eq;
			break; case Tok_plus_mod:        op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_plus_mod;
			break; case Tok_plus_mod_eq:     op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_plus_mod_eq;
			break; case Tok_minus_eq:        op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_minus_eq;
			break; case Tok_minus_mod:       op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_minus_mod;
			break; case Tok_minus_mod_eq:    op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_minus_mod_eq;
			break; case Tok_multiply_eq:     op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_multiply_eq;
			break; case Tok_multiply_mod:    op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_multiply_mod;
			break; case Tok_multiply_mod_eq: op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_multiply_mod_eq;
			break; case Tok_divide_eq:       op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_divide_eq;
			break; case Tok_mod_eq:          op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_mod_eq;
			break; case Tok_shift_left_eq:   op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_shift_left_eq;
			break; case Tok_shift_right_eq:  op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_shift_right_eq;
			break; case Tok_bitand_eq:       op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_and_eq;
			break; case Tok_bitxor_eq:       op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_xor_eq;
			break; case Tok_bitor_eq:        op_prec = 9; op_assoc = Assoc_rtl; ast_kind = Ast_or_eq;

			// Not a binary operator
			default: return lhs;
			}

			if(op_prec > precedence) {
				return lhs;
			}
			if(op_prec == precedence and op_assoc == Assoc_ltr) {
				return lhs;
			}
			++it;
			lhs = add_bop_node(ast_kind, lhs, parse_expression(op_prec));
			continue;

			__builtin_unreachable();
		} while(false);
		return lhs;
	}

	void add_decl(int name, int type, int value, bool is_mutable) {
		output.decls.emplace_back(Declaration{name, type, value, is_mutable});
	}

	AST parse() {
		COCK
		while(it != tokens.end()) {
			bool is_mutable = true;
			switch(it++->kind) {
			case Tok_fn: {
				// Top-level function declaration
				auto const name = it++;
				assert_kind(*name, Tok_identifier);
				auto const fexpr = parse_function_expr();
				add_decl(name->offset, -1, fexpr, false);
				break;
			}
			case Tok_struct: {
				// Top-level struct definition
				auto const name = it++;
				assert_kind(*name, Tok_identifier);
				auto const sexpr = parse_struct_expr();
				add_decl(name->offset, -1, sexpr, false);
				break;
			}
			case Tok_const: is_mutable = false;
			case Tok_var: {
				// Top-level declaration
				auto const name = it++;
				assert_kind(*name, Tok_identifier);
				int type = -1;
				if(it->kind == Tok_colon) {
					++it;
					type = parse_expression(DECLARATION_TYPE_PREC);
				}
				assert_kind(*it++, Tok_equals, "Expected '=' after declarator");
				int value = parse_expression();
				assert_kind(*it++, Tok_semicolon, "Expected ';' after declaration");
				add_decl(name->offset, type, value, is_mutable);
				break;
			}
			default:
				assert(false);
			}
		}
		return output;
	}

	static AST parse(std::vector<struct Token> const &tokens) {
		Parser p{{}, tokens, tokens.begin()};
		return p.parse();
	}
};

struct LocalVariable {
	int name;
	int frame_offset;
};

struct Scope {
	std::vector<LocalVariable> locals;
	Scope const *parent;
	AST const *toplevel_parent;
	int curr_stack_off = 8;

	Scope(AST const &ast) { toplevel_parent = &ast; parent = nullptr; }
	Scope(Scope const &scope) { parent = &scope; curr_stack_off = scope.curr_stack_off; }

	std::variant<int, Declaration> resolve(char const *source, int nameoff) const {
		COCK
		auto hash = keyword_hash(source + nameoff);
		for(auto const &local: locals) {
			if(keyword_hash(source + local.name) == hash) {
				return local.frame_offset;
			}
		}
		if(parent) {
			return parent->resolve(source, nameoff);
		} else {
			assert(toplevel_parent);
			for(auto const &decl: toplevel_parent->decls) {
				if(keyword_hash(source + decl.name) == hash) {
					return decl;
				}
			}
			assert(false);
		}
	}

	int alloc() {
		int retval = curr_stack_off;
		curr_stack_off += 8;
		if(curr_stack_off >= 256) assert(false);
		return retval;
	}

	int declare(int nameoff) {
		auto const result = alloc();
		locals.emplace_back(LocalVariable{nameoff, result});
		return result;
	}
};

FILE *output;

char const *farg_regs[] = {
	"rdi",
	"rsi",
	"rdx",
	"rcx",
	"r8",
	"r9",
};

void output_decl_name(char const *source, Declaration const &decl) {
	int namelen = token_source_length(source, decl.name);
	fprintf(output, "d_%.*s", namelen, source + decl.name);
}

void ref_tmplabel(int label) {
	fprintf(output, "l_%d", label);
}

int next_tmplabel = 0;

int alloc_tmplabel() {
	return next_tmplabel++;
}

int continue_label = -1;
int break_label = -1;

int make_tmplabel() {
	int result = alloc_tmplabel();
	ref_tmplabel(result);
	return result;
}

enum {
	Value,
	Ref8,
	Ref16,
	Ref32,
	Ref64,
	None,
};

int gen_node(AST const &ast, char const *source, int node_idx, Scope &scope);

void gen_value(AST const &ast, char const *source, int node_idx, Scope &scope) {
	switch(gen_node(ast, source, node_idx, scope)) {
	case Ref8:
		fprintf(output, "pop rax\nmovzx rax, byte [rax]\npush rax\n");
		return;
	case Ref16:
		fprintf(output, "pop rax\nmovzx rax, word [rax]\npush rax\n");
		return;
	case Ref32:
		fprintf(output, "pop rax\nmov eax, [rax]\npush rax\n");
		return;
	case Ref64:
		fprintf(output, "pop rax\nmov rax, [rax]\npush rax\n");
		return;
	case Value:
		return;
	case None:
		assert(false);
	}
}

void gen_none(AST const &ast, char const *source, int node_idx, Scope &scope) {
	switch(gen_node(ast, source, node_idx, scope)) {
	case Ref8:
	case Ref16:
	case Ref32:
	case Ref64:
	case Value:
		fprintf(output, "pop rax\n");
	case None:
		return;
	}
}

int gen_ref(AST const &ast, char const *source, int node_idx, Scope &scope) {
	int result = gen_node(ast, source, node_idx, scope);
	switch(result) {
	case Ref8:
	case Ref16:
	case Ref32:
	case Ref64:
		return result;
	default:
		assert(false);
	}
}

int gen_node(AST const &ast, char const *source, int node_idx, Scope &scope) {
	COCK
	auto const &node = ast.nodes.at(node_idx);
	while(true) {
		switch(node.kind) {
		case Ast_ident: {
			auto const resolved = scope.resolve(source, node.offset);
			if(std::holds_alternative<int>(resolved)) {
				int stack_offset = std::get<int>(resolved);
				fprintf(output, "lea rax, [rbp - %d]\npush rax\n", stack_offset);
				return Ref64;
			} else {
				assert(std::holds_alternative<Declaration>(resolved));
				Declaration decl = std::get<Declaration>(resolved);
				fprintf(output, "lea rax, [rel ");
				output_decl_name(source, decl);
				fprintf(output, "]\npush rax\n");
				switch(ast.eval_comptime_type_expr_size(source, decl.type)) {
				case 1: return Ref8;
				case 2: return Ref16;
				case 4: return Ref32;
				case 8: return Ref64;
				default: return Ref8; // TODO: Not be dumb
				}
			}
		}
		case Ast_int_literal: {
			int offset = node.offset;
			fprintf(output, "push %ld\n", int_literal_at_offset(source, &offset));
			return Value;
		}
		case Ast_char_literal: {
			int offset = node.offset;
			fprintf(output, "push %ld\n", char_literal_at_offset(source, &offset));
			return Value;
		}
		case Ast_string_literal: {
			int offset = node.offset;
			uint8_t tmpbuf[100];
			string_literal_at_offset(source, &offset, tmpbuf, sizeof(tmpbuf) - 1);
			fprintf(output, "[section .rodata]\n");
			int lbl = make_tmplabel();
			fprintf(output, ":\n");
			for(uint8_t const *s = tmpbuf;;) {
				fprintf(output, "db %d\n", *s);
				if(!*s++) break;
			}
			fprintf(output, "[section .text]\nlea rax, [rel ");
			ref_tmplabel(lbl);
			fprintf(output, "]\npush rax\n");
			return Ref8;
		}
		case Ast_unary_plus:
			gen_value(ast, source, node.operand, scope);
			return Value;
		case Ast_unary_minus:
			gen_value(ast, source, node.operand, scope);
			fprintf(output, "neg qword [rsp]\n");
			return Value;
		case Ast_unary_bitnot:
			gen_value(ast, source, node.operand, scope);
			fprintf(output, "not qword [rsp]\n");
			return Value;
		case Ast_unary_lognot:
			gen_value(ast, source, node.operand, scope);
			fprintf(output, "xor rcx, rcx\npop rax\ncmp rax, 0\nsete cl\npush rcx\n");
			return Value;
		case Ast_pointer_type: assert(false);
		case Ast_addr_of: {
			gen_ref(ast, source, node.operand, scope);
			return Value;
		}
		case Ast_deref:
			gen_value(ast, source, node.operand, scope);
			return Ref8; // TODO: Carry pointer type through `Value`s
		case Ast_array_type: assert(false);
		case Ast_member_access: assert(false);
		case Ast_array_subscript:
			// TODO: Multiply rhs by element size
			gen_ref(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			// Now we assume lhs is a *u8
			fprintf(output, "pop rcx\npop rax\nadd rax, rcx\npush rax\n");
			return Ref8;
		case Ast_multiply:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rax\nxor rdx, rdx\npop rcx\nmul rcx\npush rax");
			return Value;
		case Ast_divide:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rax\nxor rdx, rdx\npop rcx\ndiv rcx\npush rax");
			return Value;
		case Ast_modulus:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rax\nxor rdx, rdx\npop rcx\ndiv rcx\npush rdx");
			return Value;
		case Ast_addition: assert(false);
		case Ast_subtraction: assert(false);
		case Ast_shift_left: assert(false);
		case Ast_shift_right: assert(false);
		case Ast_bitand: assert(false);
		case Ast_bitor: assert(false);
		case Ast_bitxor: assert(false);
		case Ast_less:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rdx\npop rcx\nxor rax, rax\ncmp rcx, rdx\nsetl al\npush rax\n");
			return Value;
		case Ast_less_equal:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rdx\npop rcx\nxor rax, rax\ncmp rcx, rdx\nsetle al\npush rax\n");
			return Value;
		case Ast_greater:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rdx\npop rcx\nxor rax, rax\ncmp rcx, rdx\nsetg al\npush rax\n");
			return Value;
		case Ast_greater_equal:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rdx\npop rcx\nxor rax, rax\ncmp rcx, rdx\nsetge al\npush rax\n");
			return Value;
		case Ast_equals:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rdx\npop rcx\nxor rax, rax\ncmp rcx, rdx\nsete al\npush rax\n");
			return Value;
		case Ast_not_equal:
			gen_value(ast, source, node.lhs, scope);
			gen_value(ast, source, node.rhs, scope);
			fprintf(output, "pop rdx\npop rcx\nxor rax, rax\ncmp rcx, rdx\nsetne al\npush rax\n");
			return Value;
		case Ast_logical_and: assert(false);
		case Ast_logical_or: assert(false);
		case Ast_assign: {
			gen_value(ast, source, node.rhs, scope);
			switch(gen_ref(ast, source, node.lhs, scope)) {
			case Ref8:
				fprintf(output, "pop rax\npop rcx\nmov byte [rax], cl\n");
				return None;
			case Ref16:
				fprintf(output, "pop rax\npop rcx\nmov word [rax], cx\n");
				return None;
			case Ref32:
				fprintf(output, "pop rax\npop rcx\nmov dword [rax], ecx\n");
				return None;
			case Ref64:
				fprintf(output, "pop rax\npop rcx\nmov [rax], rcx\n");
				return None;
			}
		}
		case Ast_plus_eq:
		case Ast_plus_mod_eq: {
			gen_value(ast, source, node.rhs, scope);
			switch(gen_ref(ast, source, node.lhs, scope)) {
			case Ref8:
				fprintf(output, "pop rax\npop rcx\nadd byte [rax], cl\n");
				return None;
			case Ref16:
				fprintf(output, "pop rax\npop rcx\nadd word [rax], cx\n");
				return None;
			case Ref32:
				fprintf(output, "pop rax\npop rcx\nadd dword [rax], ecx\n");
				return None;
			case Ref64:
				fprintf(output, "pop rax\npop rcx\nadd [rax], rcx\n");
				return None;
			}
		}
		case Ast_plus_mod: assert(false);
		case Ast_minus_eq:
		case Ast_minus_mod_eq: {
			gen_value(ast, source, node.rhs, scope);
			switch(gen_ref(ast, source, node.lhs, scope)) {
			case Ref8:
				fprintf(output, "pop rax\npop rcx\nsub byte [rax], cl\n");
				return None;
			case Ref16:
				fprintf(output, "pop rax\npop rcx\nsub word [rax], cx\n");
				return None;
			case Ref32:
				fprintf(output, "pop rax\npop rcx\nsub dword [rax], ecx\n");
				return None;
			case Ref64:
				fprintf(output, "pop rax\npop rcx\nsub [rax], rcx\n");
				return None;
			}
		}
		case Ast_minus_mod:  assert(false);
		case Ast_multiply_eq: assert(false);
		case Ast_multiply_mod: assert(false);
		case Ast_multiply_mod_eq: assert(false);
		case Ast_divide_eq: assert(false);
		case Ast_mod_eq: assert(false);
		case Ast_shift_left_eq: assert(false);
		case Ast_shift_right_eq: assert(false);
		case Ast_and_eq: assert(false);
		case Ast_xor_eq: assert(false);
		case Ast_or_eq: assert(false);
		case Ast_local_const:
		case Ast_local_var: {
			gen_value(ast, source, node.rhs, scope);
			// Store it in its slot
			auto const stack_offset = scope.declare(node.next);
			fprintf(output, "pop rax\nmov [rbp - %d], rax\n", stack_offset);
			return None;
		}
		case Ast_function_expression: assert(false);
		case Ast_function_parameter: assert(false);
		case Ast_function_call: {
			auto const &callee = ast.nodes.at(node.lhs);
			assert(callee.kind == Ast_ident);
			auto hash = keyword_hash(source + callee.offset);
			switch(hash) {
			case keyword_hash("@syscall"): {
				int num_args = 0;
				int arg_node = node.next;

				while(arg_node != -1) {
					auto const &arg = ast.nodes.at(arg_node);
					gen_value(ast, source, arg.operand, scope);
					++num_args;
					arg_node = arg.next;
				}

				char const *regs[] = {
					"rax",
					"rdi",
					"rsi",
					"rdx",
					"r10",
					"r8",
					"r9",
				};

				for(; num_args;) {
					fprintf(output, "pop %s\n", regs[--num_args]);
				}
				fprintf(output, "syscall\npush rax\n");
				return Value;
			}

			default: {
				int num_args = 0;
				int arg_node = node.next;

				while(arg_node != -1) {
					auto const &arg = ast.nodes.at(arg_node);
					gen_value(ast, source, arg.operand, scope);
					++num_args;
					arg_node = arg.next;
				}
				for(; num_args;) {
					fprintf(output, "pop %s\n", farg_regs[--num_args]);
				}
				fprintf(output, "call ");

				auto const callee_ident = scope.resolve(source, callee.offset);
				if(std::holds_alternative<Declaration>(callee_ident)) {
					auto const decl = std::get<Declaration>(callee_ident);
					output_decl_name(source, decl);
					auto const fnode = ast.nodes.at(decl.value);
					assert(fnode.kind == Ast_function_expression);

					if(ast.eval_comptime_type_expr_size(source, fnode.rhs)) {
						fprintf(output, "\npush rax\n");
						return Value;
					}
					fprintf(output, "\n");
					return None;
				} else {
					assert(false);
				}
			}
			}
		}
		case Ast_function_argument: assert(false);
		case Ast_compound_statement: {
			Scope new_scope = scope;
			ASTNode curr = node;

			while(true) {
				if(curr.operand != -1) gen_none(ast, source, curr.operand, new_scope);
				if(curr.next == -1) return None;
				curr = ast.nodes.at(curr.next);
			}
		}
		case Ast_expression_statement:
			gen_none(ast, source, node.operand, scope);
			return None;
		case Ast_return_statement:
			if(node.operand != -1) {
				gen_value(ast, source, node.operand, scope);
				fprintf(output, "pop rax\n");
			}
			fprintf(output, "mov rsp, rbp\npop rbp\nret\n");
			return None;
		case Ast_if: {
			gen_value(ast, source, node.next, scope); // Check condition
			int else_label = alloc_tmplabel();
			int endif_label = alloc_tmplabel();
			fprintf(output, "pop rax\ncmp rax, 0\njz ");
			ref_tmplabel(else_label);
			fprintf(output, "\n");
			gen_none(ast, source, node.lhs, scope); // Taken
			if(node.rhs != -1) {
				fprintf(output, "jmp ");
				ref_tmplabel(endif_label);
				fprintf(output, "\n");
			}
			ref_tmplabel(else_label);
			fprintf(output, ":\n");
			if(node.rhs != -1) {
				gen_none(ast, source, node.rhs, scope); // Not taken
			}
			ref_tmplabel(endif_label);
			fprintf(output, ":\n");
			return None;
		}
		case Ast_loop: {
			int old_continue_label = continue_label;
			int old_break_label = break_label;
			continue_label = make_tmplabel();
			fprintf(output, ":\n");
			break_label = alloc_tmplabel();
			gen_none(ast, source, node.operand, scope);
			fprintf(output, "jmp ");
			ref_tmplabel(continue_label);
			fprintf(output, "\n");
			continue_label = old_continue_label;
			break_label = old_break_label;
			return None;
		}
		case Ast_switch: assert(false);
		default: assert(false);
		}
	}
}

void do_decl(AST const &ast, char const *source, Declaration const &decl) {
	COCK
	printf("Decl with name at %d, type expr %d and value %d, mutability: %d\n", decl.name, decl.type, decl.value, decl.is_mutable);
	auto const &node = ast.nodes.at(decl.value);
	switch(node.kind) {
	case Ast_function_expression: {
		fprintf(output, "[section .text]\n");
		output_decl_name(source, decl);
		fprintf(output, ":\n");
		Scope fscope = ast;
		fprintf(output, "push rbp\nmov rbp, rsp\n");
		// Push arguments on the stack and declare them
		int num_args = 0;
		int arg_idx = node.lhs;
		while(arg_idx != -1) {
			auto const &arg = ast.nodes.at(arg_idx);
			assert(arg.lhs != -1);
			assert(ast.eval_comptime_type_expr_size(source, arg.rhs) <= 8);
			fscope.declare(arg.lhs);
			fprintf(output, "push %s\n", farg_regs[num_args++]);
			arg_idx = arg.next;
		}
		// This ought to be enough local variables for anybody
		fprintf(output, "sub rsp, 256\n");
		gen_none(ast, source, node.next, fscope);

		// If you didn't return we'll just leave without returning a value
		fprintf(output, "mov rsp, rbp\npop rbp\nret\n");
		return;
	}
	default:
		if(decl.is_mutable) {
			fprintf(output, "[section .data]\n");
		} else {
			fprintf(output, "[section .rodata]\n");
		}
		output_decl_name(source, decl);
		fprintf(output, ":\n");

		int type_size;
		if(decl.type == -1) {
			assert(!"Todo: Inferred types");
		} else {
			type_size = ast.eval_comptime_type_expr_size(source, decl.type);
		}
		uint64_t decl_value = ast.eval_comptime_expr(source, decl.value);
		if(decl_value == 0) {
			fprintf(output, "times %d db 0\n", type_size);
		} else {
			for(int i = 0; i < type_size; ++i) {
				if(i < sizeof(uint64_t)) {
					fprintf(output, "db %d\n", ((uint8_t*)&decl_value)[i]);
				} else {
					fprintf(output, "db 0\n");
				}
			}
		}
		return;
	}
}

int main(int argc, char const **argv) {
	assert(argc == 3);
	output = fopen(argv[2], "w");
	int fd = open(argv[1], O_RDONLY);
	int file_len = lseek(fd, 0, SEEK_END);
	auto source_bytes = (char const *)mmap(NULL, file_len + 0x1000, PROT_READ, MAP_PRIVATE, fd, 0);
	auto const tokens = tokenize(source_bytes);
	auto const ast = Parser::parse(tokens);
	for(auto const &decl: ast.decls) {
		do_decl(ast, source_bytes, decl);
	}
	for(int i = 0; i < ast.nodes.size(); ++ i) {
		auto const &node = ast.nodes[i];
		printf("Node %d kind %d, next %d, operands %d %d\n", i, node.kind, node.next, node.lhs, node.rhs);
	}
	fprintf(output,
		"[global _start]\n"
		"_start:\n"
		"mov rdi, [rsp]\n"
		"lea rsi, [rsp + 8]\n"
		"lea rdx, [rsp + rdi * 8 + 0x10]\n"
		"call d_main\n"
		"mov rdi, rax\n"
		"push rdi\n"
		"call d_atexit_hook\n"
		"pop rdi\n"
		"mov rax, 60\n"
		"syscall\n"
	);
}
