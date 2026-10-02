"""
Chat and Conversational Text Normalizer for NLP & GED.
Translates informal mobile/chat shorthand, fat-finger keystrokes, missing apostrophes,
token mergers, and conversational subject-verb discord into grammatically sound standard English.
"""
import re
from typing import Tuple, List, Dict


# Common chat shortcuts and phonetic slang
CHAT_SHORTCUTS: Dict[str, str] = {
    "u": "you",
    "r": "are",
    "pls": "please",
    "plz": "please",
    "plzz": "please",
    "plzzz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "bcoz": "because",
    "bcz": "because",
    "becoz": "because",
    "becos": "because",
    "cuz": "because",
    "coz": "because",
    "cos": "because",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "lemme": "let me",
    "kinda": "kind of",
    "idk": "I do not know",
    "btw": "by the way",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "omg": "oh my god",
    "tbh": "to be honest",
    "rn": "right now",
    "asap": "as soon as possible",
    "dm": "direct message",
    "pic": "picture",
    "pics": "pictures",
    "info": "information",
    "sec": "second",
    "min": "minute",
    "abt": "about",
    "sry": "sorry",
    "srry": "sorry",
    "gud": "good",
    "gr8": "great",
    "msg": "message",
    "msgs": "messages",
    "txt": "text",
    "diff": "different",
    "prob": "problem",
    "bday": "birthday",
    "frnd": "friend",
    "frnds": "friends",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "bro": "brother",
    "sis": "sister",
}

# Missing contraction apostrophes
CONTRACTION_MAP: Dict[str, str] = {
    "dont": "don't",
    "cant": "can't",
    "wont": "won't",
    "didnt": "didn't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
    "im": "I'm",
    "youre": "you're",
    "theyre": "they're",
    "weve": "we've",
    "theyve": "they've",
    "youve": "you've",
    "thats": "that's",
    "whats": "what's",
    "theres": "there's",
    "heres": "here's",
}

# Run-on merged chat words
MERGED_WORDS: Dict[str, str] = {
    "alot": "a lot",
    "alotof": "a lot of",
    "iama": "I am a",
    "iam": "I am",
    "iamgoing": "I am going",
    "infront": "in front",
    "infrontof": "in front of",
    "atleast": "at least",
    "aswell": "as well",
    "aswellas": "as well as",
    "thankyou": "thank you",
    "eachother": "each other",
    "outof": "out of",
    "nevermind": "never mind",
    "everytime": "every time",
    "allright": "all right",
    "tothe": "to the",
    "onthe": "on the",
    "inthe": "in the",
    "atthe": "at the",
    "bythe": "by the",
    "forthe": "for the",
    "fromthe": "from the",
    "withthe": "with the",
    "allthe": "all the",
    "havea": "have a",
    "wasa": "was a",
    "witha": "with a",
    "fora": "for a",
    "takea": "take a",
    "makea": "make a",
    "likea": "like a",
    "justa": "just a",
    "didnot": "did not",
    "donot": "do not",
    "doesnot": "does not",
    "couldnot": "could not",
    "shouldnot": "should not",
    "wouldnot": "would not",
    "havebeen": "have been",
    "hasbeen": "has been",
    "hadbeen": "had been",
    "willbe": "will be",
    "wouldbe": "would be",
    "couldbe": "could be",
    "shouldbe": "should be",
    "mustbe": "must be",
    "wantto": "want to",
    "needto": "need to",
    "haveto": "have to",
    "hasto": "has to",
    "hadto": "had to",
    "ableto": "able to",
    "usedto": "used to",
    "goingto": "going to",
    "tryingto": "trying to",
    "lookat": "look at",
    "talkto": "talk to",
    "waitfor": "wait for",
    "listento": "listen to",
    "athome": "at home",
    "atwork": "at work",
    "atschool": "at school",
    "allof": "all of",
    "oneof": "one of",
    "someof": "some of",
    "mostof": "most of",
    "partof": "part of",
    "becauseof": "because of",
    "somany": "so many",
    "toomuch": "too much",
    "toomany": "too many",
    "verymuch": "very much",
    "verygood": "very good",
    "goodmorning": "good morning",
    "goodnight": "good night",
    "goodafternoon": "good afternoon",
    "lastnight": "last night",
    "lastweek": "last week",
    "lastyear": "last year",
    "nextweek": "next week",
    "nextyear": "next year",
    "thisweek": "this week",
    "thisyear": "this year",
    "withyou": "with you",
    "formy": "for my",
    "howareyou": "how are you",
    "whatareyou": "what are you",
    "whereareyou": "where are you",
    "youare": "you are",
    "theyare": "they are",
    "weare": "we are",
    "heis": "he is",
    "sheis": "she is",
    "itis": "it is",
    "thereis": "there is",
    "thereare": "there are",
    "thisis": "this is",
    "thatis": "that is",
}

# Conversational typos (including user's chat patterns)
CONVERSATIONAL_TYPOS: Dict[str, str] = {
    "mutiple": "multiple",
    "parameteres": "parameters",
    "vocaolobary": "vocabulary",
    "vocabularly": "vocabulary",
    "synimys": "synonyms",
    "synonims": "synonyms",
    "undersating": "understanding",
    "setneces": "sentences",
    "sentenes": "sentences",
    "speeling": "spelling",
    "speling": "spelling",
    "mistak": "mistake",
    "mistaks": "mistakes",
    "freinds": "friends",
    "togther": "together",
    "definatly": "definitely",
    "definately": "definitely",
    "tommorow": "tomorrow",
}


class ChatNormalizer:
    """
    Normalizes conversational, chat, and informal social media English
    into standard English while tracking exact modifications.
    """

    @classmethod
    def normalize(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Normalizes chat shortcuts, missing apostrophes, elongated chars, and conversational typos.
        """
        revised = text
        changes = []

        # 0. Missing space after punctuation and capitalize new sentence after ?, !, and .
        revised = re.sub(r'([A-Za-z])([!?])\s*([a-z])', lambda m: f'{m.group(1)}{m.group(2)} {m.group(3).upper()}', revised)
        revised = re.sub(r'([A-Za-z])([,;])([A-Za-z])', r'\1\2 \3', revised)
        revised = re.sub(r'([a-z]{2,})\.([A-Za-z])', lambda m: f'{m.group(1)}. {m.group(2).upper()}', revised)

        # 1. Compress elongated characters (e.g. soooo -> so, pleaaase -> please, heyyy -> hey)
        def compress_repeats(m):
            w = m.group(0)
            # Replace 3 or more repeated letters with 1 (or 2 if common e.g. 'too', 'see')
            compressed = re.sub(r'(.)\1{2,}', r'\1', w)
            return compressed

        revised = re.sub(r'\b[a-zA-Z]+\b', compress_repeats, revised)

        # 2. 'i ama trying' -> 'I am trying' / 'i ama' -> 'I am a'
        if re.search(r'\b(i|I)\s+ama\s+trying\b', revised, re.I):
            revised = re.sub(r'\b(i|I)\s+ama\s+trying\b', 'I am trying', revised, flags=re.I)
            changes.append({"original": "i ama trying", "normalized": "I am trying", "type": "Chat Slang / Grammar"})
        elif re.search(r'\b(i|I)\s+ama\b', revised, re.I):
            revised = re.sub(r'\b(i|I)\s+ama\b', 'I am a', revised, flags=re.I)
            changes.append({"original": "i ama", "normalized": "I am a", "type": "Chat Slang / Grammar"})

        # 3. Merged chat tokens & Dynamic Subword Segmentation
        for merged, target in MERGED_WORDS.items():
            pattern = rf'\b{merged}\b'
            if re.search(pattern, revised, re.I):
                revised = re.sub(pattern, target, revised, flags=re.I)
                changes.append({"original": merged, "normalized": target, "type": "Merged Word"})

        # Dynamic Subword Segmentation for unrecognized run-on words (e.g. speelingmistake, verygood)
        try:
            from src.model.segmenter import SubwordSegmenter
            def segment_repl(m):
                token = m.group(0)
                split = SubwordSegmenter.split_compound(token)
                if split:
                    changes.append({"original": token, "normalized": split, "type": "Merged Word"})
                    return split
                return token

            revised = re.sub(r'\b[A-Za-z]{4,}\b', segment_repl, revised)
        except Exception:
            pass

        # 4. Conversational fat-finger typos
        for typo, correct in CONVERSATIONAL_TYPOS.items():
            pattern = rf'\b{typo}\b'
            if re.search(pattern, revised, re.I):
                revised = re.sub(pattern, correct, revised, flags=re.I)
                changes.append({"original": typo, "normalized": correct, "type": "Spelling / Typo"})

        # 5. Missing contraction apostrophes (dont -> don't, cant -> can't, im -> I'm)
        for cont, full in CONTRACTION_MAP.items():
            pattern = rf'\b{cont}\b'
            if re.search(pattern, revised, re.I):
                # Check case preservation
                repl = full
                revised = re.sub(pattern, repl, revised, flags=re.I)
                changes.append({"original": cont, "normalized": full, "type": "Missing Apostrophe"})

        # 6. Chat shortcuts (u -> you, pls -> please, bcoz -> because, etc.)
        for shortcut, standard in CHAT_SHORTCUTS.items():
            pattern = rf'\b{shortcut}\b'
            if re.search(pattern, revised, re.I):
                revised = re.sub(pattern, standard, revised, flags=re.I)
                changes.append({"original": shortcut, "normalized": standard, "type": "Chat Slang"})

        # 7. 'ur' disambiguation: 'ur' before noun -> 'your', before adj/participle -> "you're"
        revised = re.sub(r'\bur\s+([a-z]+ing|[a-z]+ed|right|welcome|good|ready|late)\b', r"you're \1", revised, flags=re.I)
        revised = re.sub(r'\bur\s+([a-z]+)\b', r'your \1', revised, flags=re.I)

        # 8. 1st-person SVA in chat: "if I chats here" -> "if I chat here", "I runs" -> "I run"
        revised = re.sub(r'\b(if\s+)?(i|I)\s+chats\b', r'\1I chat', revised)
        revised = re.sub(r'\b(i|I)\s+(walks|runs|eats|knows|likes|wants|thinks|sees|says)\b',
                         lambda m: f"I {m.group(2)[:-1]}", revised)

        # 9. Capitalize standalone 'i' pronoun
        revised = re.sub(r'\bi\b', 'I', revised)

        return revised, changes
