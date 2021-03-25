import re

class BVHScanner():
    """
        Class used to scan bvh files using regex
    """

    def __init__(self):
        self.scanner = re.Scanner([
            (r"[a-zA-Z_]\w*", self.s_ident),
            (r"-*[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", self.s_digit),
            (r"{", self.s_opening_brace),
            (r"}", self.s_closing_brace),
            (r"\\n", self.s_end_line),
            (r":", None),
            (r"\s+", None)
        ])

    def scan(self, path):
        with open(path, "r") as file:
            raw_content = file.read()

        hierarchy, motion = raw_content.split("MOTION")
        tokens_hierarchy, _ = self.scanner.scan(hierarchy)
        return tokens_hierarchy, motion 

    def s_ident(self, scanner, token): 
        return 'IDENTIFIER', token

    def s_digit(self, scanner, token): 
        return 'DIGIT', token

    def s_opening_brace(self, scanner, token):
        return 'OPEN', token

    def s_closing_brace(self, scanner, token):
        return 'CLOSE', token

    def s_end_line(self, scanner, token):
        return 'END_LINE', token