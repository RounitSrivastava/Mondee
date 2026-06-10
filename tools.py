def calculator(expression):
    try:
        result = eval(expression)
        return result
    except:
        return "Invalid Expression"


def web_search(query):
    return f"Searching for: {query}"