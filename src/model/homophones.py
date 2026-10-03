"""
Contextual Homophone and Confused Word Disambiguation Engine.
Detects commonly confused English word pairs:
accept/except, affect/effect, their/there/they're, your/you're, its/it's,
then/than, to/too, lose/loose, advice/advise, weather/whether, etc.
"""
import re
from typing import Tuple, List, Dict


CONFUSED_WORD_RULES = [
    # 1. accept vs except
    (
        r'\b(gift|letter|package|everything|everyone|everybody|nothing|nobody|all|anything|anyone|anybody|none|each|card|person|man|woman|book|day|night)\s+accept\s+(for\b|the\b|a\b|an\b|me\b|him\b|her\b|us\b|them\b|my\b|your\b|his\b)',
        r'\1 except \2',
        "Incorrect word 'accept' (verb meaning receive). The preposition 'except' (meaning excluding) should be used."
    ),
    (
        r'\baccept\s+for\b',
        'except for',
        "Confused word: 'accept' is a verb. Use the preposition 'except for' to denote exclusion."
    ),
    (
        r'\bexcept\s+(the\s+(?:gift|invitation|award|offer|challenge|terms|job|money|package))\b',
        r'accept \1',
        "Confused word: 'except' means exclude. Use 'accept' (to receive willingly)."
    ),

    # 2. then vs than (comparatives)
    (
        r'\b(better|more|less|faster|slower|bigger|smaller|greater|worse|rather|other|higher|lower|older|younger|taller|shorter|easier|harder|stronger|longer|shorter)\s+then\b',
        r'\1 than',
        "Use 'than' for comparisons. 'Then' indicates time or sequence."
    ),

    # 3. affect vs effect
    (
        r'\b(the|an|a|any|no|side|adverse|profound|direct|indirect|lasting)\s+([a-z]+\s+)?affect\b',
        lambda m: m.group(0).replace('affect', 'effect').replace('Affect', 'Effect'),
        "'Effect' is typically a noun (a result), whereas 'affect' is usually a verb (to influence)."
    ),
    (
        r'\b(will|would|can|could|to|did|does|do|may|might|must|should)\s+effect\b',
        r'\1 affect',
        "'Affect' is typically the verb meaning to influence or change."
    ),
    (
        r'\b(deeply|greatly|severely|badly|adversely|negatively|strongly|directly|indirectly)\s+effected\b',
        r'\1 affected',
        "'Affected' is the verb form meaning influenced or altered. 'Effected' means brought about."
    ),
    (
        r'\b(was|were|is|are|am|been|being)\s+(?:(\w+ly)\s+)?effected\b',
        lambda m: m.group(0).replace('effected', 'affected').replace('Effected', 'Affected'),
        "Passive voice requiring the verb 'affected' (influenced), not noun-derived 'effected'."
    ),
    (
        r'\btake\s+affect\b',
        'take effect',
        "Idiomatic expression is 'take effect' (meaning to become active)."
    ),

    # 4. their vs there vs they're
    (
        r'\btheir\s+(is|are|was|were)\b',
        r'there \1',
        "Use existential pronoun 'there' with verbs of being (there is/are/was/were), not possessive 'their'."
    ),
    (
        r'\btheir\s+(going|coming|doing|having|leaving|making|getting|planning|working|trying|saying|thinking)\b',
        r"they're \1",
        "Confused homophone: use contraction 'they're' (they are), not possessive 'their'."
    ),
    (
        r'\bover\s+their\b',
        'over there',
        "Use 'there' to refer to a place or location, not possessive 'their'."
    ),
    (
        r'\bright\s+their\b',
        'right there',
        "Use 'there' to specify location ('right there')."
    ),
    (
        r'\bthere\s+(home|house|car|room|school|parents|friends|backpack|country|village|town|shoes)\b',
        r'their \1',
        "Use possessive pronoun 'their' before a noun, not adverb 'there'."
    ),

    # 4b. siting vs sitting
    (
        r'\b(was|were|is|are|am|been|being)\s+siting\b',
        r'\1 sitting',
        "Contextual spelling / confused word: Use 'sitting' (from verb 'to sit')."
    ),
    (
        r'\bsiting\s+(in|on|at|down|there|here|next|behind|beside|near|by)\b',
        r'sitting \1',
        "Contextual spelling / confused word: In locative posture contexts, use 'sitting' instead of rare homograph 'siting'."
    ),

    # 5. your vs you're
    (
        r'\byour\s+(welcome|going|doing|coming|the\s+best|right|wrong|leaving|making|ready|redy|sure|able|happy|glad)\b',
        r"you're \1",
        "Use contraction 'you're' (you are) before adjectives, participles, or in 'you're welcome'."
    ),
    (
        r'\byou\'re\s+([a-z]+\s+)?(backpack|bag|car|dog|cat|house|phone|book|friend|mom|dad|family|name|computer|keys|shoes|ticket|clothes)\b',
        r'your \1\2',
        "Use possessive pronoun 'your' before a noun phrase, not contraction 'you're'."
    ),

    # 6. its vs it's
    (
        r'\bits\s+(a|an|the|very|not|going|been|hard|easy|time|okay|clear|important|raining|snowing|cold|hot|getting|coming|working)\b',
        r"it's \1",
        "Use contraction 'it's' (it is / it has) when followed by verbs, articles, or adjectives."
    ),
    (
        r"\bit's\s+(?!a\b|an\b|the\b|not\b|been\b|going\b|raining\b|snowing\b|cold\b|hot\b|coming\b|getting\b|working\b|taking\b|doing\b|happening\b|time\b|hard\b|easy\b|good\b|bad\b|clear\b|true\b|false\b|okay\b|ok\b|better\b|worse\b|best\b|like\b|just\b|already\b|still\b)([a-z]+)\b",
        r'its \1',
        "Use possessive pronoun 'its' (without apostrophe) before a noun. 'It's' is a contraction for 'it is'."
    ),

    # 7. to vs too
    (
        r'\bto\s+(much|many|late|early|fast|slow|bad|good|hard|easy|far|close|hot|cold|expensive|cheap)\b',
        r'too \1',
        "Use 'too' (meaning excessively or also) before adjectives or adverbs."
    ),
    (
        r'\bme\s+to(?=[,\.!\?;\)]|\s*$)',
        'me too',
        "Use 'too' meaning 'also' in 'me too'."
    ),

    # 8. lose vs loose
    (
        r'\b(to|will|don\'t|didn\'t|might|can)\s+loose\b',
        r'\1 lose',
        "'Lose' is a verb meaning to misplace or be defeated. 'Loose' is an adjective meaning not tight."
    ),
    (
        r'\bloose\s+(weight|the\s+game|my|your|his|her|their|control|hope|time|money|focus|patience|ground)\b',
        r'lose \1',
        "Use verb 'lose' (to suffer loss), not adjective 'loose'."
    ),

    # 9. advice vs advise
    (
        r'\b(give|gave|giving|need|some|any|a\s+piece\s+of|good|bad)\s+advise\b',
        r'\1 advice',
        "'Advice' is a noun (recommendation). 'Advise' is a verb (to give advice)."
    ),
    (
        r'\b(i|we|they|he|she)\s+advice\b',
        r'\1 advise',
        "Use verb 'advise' when denoting the action of counseling."
    ),
    (
        r'\b(the\s+)?(teacher|doctor|father|mother|boss|friend|counselor|lawyer)\s+advice\s+(me|him|her|us|them)\b',
        r'\1\2 advised \3',
        "Use verb form 'advised' when describing the past action of counseling."
    ),
    (
        r'\badvice\s+(me|him|her|us|them)\s+to\b',
        r'advised \1 to',
        "Use verb 'advised' (counseled) followed by an infinitive."
    ),

    # 10. weather vs whether
    (
        r'\bweather\s+or\s+not\b',
        'whether or not',
        "Use conjunction 'whether' (introducing alternatives), not noun 'weather' (climate)."
    ),
    (
        r'\bwether\s+or\s+not\b',
        'whether or not',
        "Corrected typo 'wether' to conjunction 'whether'."
    ),
    (
        r'\b(wonder|wondered|wondering|ask|asked|know|knew|knows|decide|decided|doubt|doubted|see|determine|check|evaluate|evaluating|evaluated|consider|considered|investigate)\s+(weather|wether)\b',
        r'\1 whether',
        "Use conjunction 'whether' after verbs of inquiry or uncertainty."
    ),
    (
        r'\b(weather|wether)\s+(it|he|she|they|we|you|i)\s+(would|could|should|can|will|is|was|were|might)\b',
        r'whether \1 \2',
        "Use conjunction 'whether' introducing a subordinate clause, not noun 'weather'."
    ),
    (
        r'\b(weather|wether)\s+(the|this|that|these|those)\b',
        r'whether \1',
        "Use conjunction 'whether' introducing a subordinate clause before a noun phrase."
    ),
    (
        r'\b(cold|hot|warm|rainy|sunny|bad|nice|good|stormy|chilly)\s+(whether|wether)\b',
        r'\1 weather',
        "Use noun 'weather' to denote climatic conditions."
    ),

    # 11. form vs from (Real-Word Spelling Mistake)
    (
        r'\bform\s+(my|his|her|their|there|our|the|a|an|him|her|them|us|me|school|home|London|work|scratch)\b',
        r'from \1',
        "Real-word spelling error: 'form' was used instead of preposition 'from'."
    ),
    (
        r'\b(come|came|coming|comes|away|far|different|differ|stems|stemming|apart|separate|originate|originates)\s+form\b',
        r'\1 from',
        "Use preposition 'from' to indicate origin or distance, not noun/verb 'form'."
    ),
    (
        r'\b(fill|filling|filled|submit|submitted|submitting|application|registration|entry)\s+from\b',
        r'\1 form',
        "Use noun 'form' (document) in this context, not preposition 'from'."
    ),

    # 12. quiet vs quite (Real-Word Spelling Mistake)
    (
        r'\bquiet\s+(a\b|an\b|good\b|well\b|sure\b|right\b|different\b|simple\b|easy\b|hard\b|fast\b|slow\b|happy\b|sad\b|big\b|small\b|high\b|low\b|clear\b|often\b|recently\b|few\b|studious\b|impressive\b|satisfied\b|content\b|capable\b|ready\b|likely\b|certain\b|remarkable\b)',
        r'quite \1',
        "Real-word spelling mistake: 'quite' (meaning completely or fairly) should be used instead of 'quiet' (silent)."
    ),
    (
        r'\b(was|were|is|are|am)\s+quiet\s+([a-z]+(?:ous|ful|ive|able|ible|ed|ing|ic|al|ent|ant))\b',
        r'\1 quite \2',
        "Real-word spelling mistake: use adverb 'quite' before an adjective."
    ),
    (
        r'\b(keep|kept|stay|stayed|remain|remained|be)\s+quite\b',
        r'\1 quiet',
        "Use adjective 'quiet' (meaning calm or silent), not adverb 'quite'."
    ),
    (
        r'\b(a|very)\s+quite\s+(place|room|night|person|boy|girl|corner|environment|town|village|house)\b',
        r'\1 quiet \2',
        "Use adjective 'quiet' (free from noise), not adverb 'quite'."
    ),

    # 13. peace vs piece (Homophone / Real-Word Error)
    (
        r'\bpeace\s+of\s+(cake|paper|bread|land|pizza|wood|metal|cloth|meat|chalk|evidence|information|advice|furniture|art)\b',
        r'piece of \1',
        "Confused word: 'piece' is a portion or section. 'Peace' means tranquility or absence of conflict."
    ),
    (
        r'\b(a|another|one|small|large|little|big)\s+peace\s+of\b',
        r'\1 piece of',
        "Use 'piece' (a portion), not 'peace' (absence of war/conflict)."
    ),
    (
        r'\brest\s+in\s+piece\b',
        'rest in peace',
        "Idiom is 'rest in peace' (meaning tranquility in death)."
    ),
    (
        r'\b(bring|brings|brought|find|found|live\s+in|wants?|seeking|make|made)\s+piece\b',
        r'\1 peace',
        "Use 'peace' (harmony/tranquility), not 'piece' (a portion)."
    ),
    (
        r'\bpiece\s+and\s+(solace|harmony|quiet|quietness|tranquility|calm|prosperity|order)\b',
        r'peace and \1',
        "Use 'peace' (state of tranquility) in binomial pairs."
    ),
    (
        r'\bworld\s+piece\b',
        'world peace',
        "Use 'peace' (state of harmony/no conflict) in 'world peace'."
    ),

    # 14. break vs brake (Homophone / Real-Word Error)
    (
        r'\b(car|truck|bus|vehicle|bicycle|bike|train)\s+(breaks|braked|breaking)\b',
        r'\1 brakes',
        "Use 'brake' (device for stopping a vehicle), not 'break' (to fracture or pause)."
    ),
    (
        r'\b(hit|steps?|stepped|press|pressed|applied|apply|slams?|slammed)\s+(?:on\s+)?the\s+breaks\b',
        r'\1 on the brakes',
        "Use 'brakes' for stopping a vehicle, not 'breaks'."
    ),
    (
        r'\btake\s+a\s+brake\b',
        'take a break',
        "Idiom is 'take a break' (rest or intermission), not 'brake'."
    ),
    (
        r'\b(lunch|coffee|short|spring|summer|winter)\s+brake\b',
        r'\1 break',
        "Use 'break' (a period of rest), not 'brake' (stopping mechanism)."
    ),
    (
        r'\bgive\s+me\s+a\s+brake\b',
        'give me a break',
        "Idiom is 'give me a break'."
    ),

    # 15. desert vs dessert (Real-Word Spelling Error)
    (
        r'\b(sweet|chocolate|delicious|tasty|ice\s+cream|pudding|after\s+dinner)\s+([a-z]+\s+)?desert\b',
        r'\1 \2dessert',
        "'Dessert' (with double 's') is a sweet dish served after a meal. 'Desert' is an arid, sandy land."
    ),
    (
        r'\b(eat|ate|eating|ordered|ordering|order|have|had|for|as)\s+desert\b',
        r'\1 dessert',
        "Use 'dessert' (sweet course), not 'desert' (arid land)."
    ),
    (
        r'\b(Sahara|dry|arid|hot|sandy|Mojave|Gobi)\s+dessert\b',
        r'\1 desert',
        "Use 'desert' (arid land), not 'dessert' (sweet food)."
    ),

    # 16. hear vs here (Homophone / Real-Word Error)
    (
        r'\b(can|could|did|will|would)\s+(?:you|we|they|he|she|i)\s+here\b',
        lambda m: m.group(0).replace('here', 'hear').replace('Here', 'Hear'),
        "Use verb 'hear' (to perceive sound) after modal auxiliary."
    ),
    (
        r'\b(not|cannot|couldn\'?t|can\'?t|did\s+not|didn\'?t)\s+here\b',
        r'\1 hear',
        "Use verb 'hear' (to perceive sound), not adverb 'here'."
    ),
    (
        r'\b(can|could|did|will)\s+here\s+(me|you|it|the|that|music|voice|sound|anything)\b',
        r'\1 hear \2',
        "Use verb 'hear' (to perceive sound), not adverb 'here' (in this place)."
    ),
    (
        r'\bhere\s+(the\s+(?:sound|music|voice|noise|bell|alarm|whisper|song))\b',
        r'hear \1',
        "Use verb 'hear' (to listen/perceive sound), not 'here'."
    ),
    (
        r'\b(come|stand|sit|stay|wait|live|over|right)\s+hear\b',
        r'\1 here',
        "Use adverb 'here' (this location), not verb 'hear'."
    ),
    (
        r'\bhear\s+is\s+the\b',
        'here is the',
        "Use 'here is the' to indicate location or presentation."
    ),

    # 17. bare vs bear (Homophone / Real-Word Error)
    (
        r'\b(cannot|can\'?t|could\s+not|couldn\'?t|hard\s+to|able\s+to)\s+bare\b',
        r'\1 bear',
        "Use verb 'bear' (to endure/tolerate), not adjective 'bare' (uncovered)."
    ),
    (
        r'\bbare\s+with\s+me\b',
        'bear with me',
        "The idiom is 'bear with me' (meaning please be patient)."
    ),
    (
        r'\b(with\s+(?:my|his|her|their|our))\s+bear\s+(hands|feet)\b',
        r'\1 bare \2',
        "Use adjective 'bare' (naked or uncovered), not 'bear'."
    ),

    # 18. dairy vs diary (Real-Word Spelling Error)
    (
        r'\b(secret|personal|daily|private)\s+dairy\b',
        r'\1 diary',
        "Use noun 'diary' (personal journal), not 'dairy'."
    ),
    (
        r'\bin\s+(my|his|her|their|our|a)\s+dairy\b',
        r'in \1 diary',
        "Use noun 'diary' (personal journal), not 'dairy'."
    ),
    (
        r'\b(wrote|write|writing|written|keep|dear|secret|personal)\s+(in\s+)?(my|his|her|their|our)\s+dairy\b',
        lambda m: m.group(0).replace('dairy', 'diary').replace('Dairy', 'Diary'),
        "Use noun 'diary' (daily personal record), not 'dairy' (milk products)."
    ),
    (
        r'\ba\s+personal\s+dairy\b',
        'a personal diary',
        "Use 'diary' (personal journal), not 'dairy'."
    ),

    # 19. angle vs angel (Real-Word Spelling Error)
    (
        r'\b(guardian|fallen|sweet|beautiful|heavenly)\s+angle\b',
        r'\1 angel',
        "Use noun 'angel' (spiritual being / kind person), not 'angle' (geometric corner)."
    ),
    (
        r'\blooked\s+like\s+an\s+angle\b',
        'looked like an angel',
        "Use 'angel' (heavenly being), not 'angle'."
    ),
    (
        r'\b(acute|obtuse|right|degrees?|triangle)\s+angel\b',
        r'\1 angle',
        "Use 'angle' (geometric measurement), not 'angel'."
    ),

    # 20. trial vs trail (Real-Word Spelling Error)
    (
        r'\b(hiking|mountain|forest|walking|nature|dirt|bike)\s+trial\b',
        r'\1 trail',
        "Use 'trail' (path or track), not 'trial' (court proceeding or test)."
    ),
    (
        r'\b(court|jury|murder|criminal|clinical|fair)\s+trail\b',
        r'\1 trial',
        "Use 'trial' (judicial proceeding/test), not 'trail' (path)."
    ),
    (
        r'\btrail\s+and\s+error\b',
        'trial and error',
        "Idiom is 'trial and error'."
    ),

    # 21. principal vs principle (Homophone / Real-Word Error)
    (
        r'\b((?:advanced\s+)?(?:mathematical|scientific|economic|theoretical|physics|accounting|engineering|philosophical))\s+principals?\b',
        r'\1 principles',
        "Contextual Word Choice: 'principles' (fundamental rules or foundations of a discipline) should be used instead of 'principal'."
    ),
    (
        r'\b(school|high\s+school|college)\s+principle\b',
        r'\1 principal',
        "Use 'principal' (head of a school), not 'principle' (fundamental truth/rule)."
    ),
    (
        r'\b(fundamental|basic|moral|ethical|guiding|scientific|core|underlying|foundational)\s+principals\b',
        r'\1 principles',
        "Use 'principles' (fundamental truths/laws), not 'principals'."
    ),
    (
        r'\b(fundamental|basic|moral|ethical|guiding|scientific|core|underlying|foundational)\s+principal\b',
        r'\1 principle',
        "Use 'principle' (moral rule/law), not 'principal'."
    ),
    (
        r'\bprinciple\s+reason\b',
        'principal reason',
        "Use 'principal' (meaning main or primary), not 'principle'."
    ),

    # 22. coarse vs course (Homophone / Real-Word Error)
    (
        r'\bof\s+coarse\b(?!\s+(?:sand|salt|grain|hair|fabric|texture|ground|cloth|material|powder|thread))\b',
        'of course',
        "Idiom is 'of course' (meaning naturally/certainly), not 'coarse' (rough)."
    ),
    (
        r'\b(golf|main|crash|online|training)\s+coarse\b',
        r'\1 course',
        "Use 'course' (a path, ground, or study program), not 'coarse' (rough)."
    ),

    # 23. aloud vs allowed (Homophone / Real-Word Error)
    (
        r'\b(not|never|aren\'t|isn\'t|wasn\'t|weren\'t)\s+aloud\s+to\b',
        r'\1 allowed to',
        "Use past participle 'allowed' (permitted), not adverb 'aloud' (audibly)."
    ),
    (
        r'\bread\s+allowed\b',
        'read aloud',
        "Use 'read aloud' (audibly, out loud), not 'allowed'."
    ),

    # 24. breath vs breathe (Real-Word Spelling Error)
    (
        r'\b(take|took|taking|catch|hold)\s+(a\s+|my\s+|your\s+|his\s+|her\s+)?breathe\b',
        r'\1 \2breath',
        "Use noun 'breath' (air inhaled/exhaled). 'Breathe' is the verb action."
    ),
    (
        r'\bdeep\s+breathe\b',
        'deep breath',
        "Use noun 'breath' in 'deep breath'."
    ),
    (
        r'\b(cannot|can\'t|struggled\s+to|hard\s+to)\s+breath\b',
        r'\1 breathe',
        "Use verb 'breathe' for the act of inhaling and exhaling."
    ),

    # 25. cloths vs clothes (Real-Word Spelling Error)
    (
        r'\b(wearing|wear|put\s+on|clean|new|dirty|bought|my|his|her|their|your|our)\s+cloths\b',
        r'\1 clothes',
        "Use 'clothes' (garments to wear), not 'cloths' (pieces of fabric/rags)."
    ),

    # 26. sight vs site vs cite (Homophone / Real-Word Error)
    (
        r'\b(web|construction|camping|historic|building)\s+sight\b',
        r'\1 site',
        "Use 'site' (location or website), not 'sight' (vision)."
    ),
    (
        r'\b(out\s+of|lost\s+his|lost\s+her|in\s+plain)\s+site\b',
        r'\1 sight',
        "Use 'sight' (visual perception), not 'site' (place)."
    ),

    # 27. passed vs past
    (
        r'\bin\s+the\s+passed\b',
        'in the past',
        "Use noun 'past' to denote previous time, not verb 'passed'."
    ),

    # 28. stationary vs stationery
    (
        r'\b(office|writing|paper)\s+stationary\b',
        r'\1 stationery',
        "Writing materials are 'stationery' (with an 'e'). 'Stationary' means not moving."
    ),

    # 29. Double Negatives
    (
        r'\b(don\'?t|didn\'?t|can\'?t|couldn\'?t|haven\'?t|hasn\'?t|won\'?t)\s+(\w+\s+)?(no|nothing|nobody|nowhere|no\s+one)\b',
        lambda m: m.group(0).replace('nothing', 'anything').replace('nobody', 'anybody').replace('nowhere', 'anywhere').replace('no one', 'anyone').replace('no ', 'any '),
        "Double negative: English standard grammar requires 'any/anything/anybody' after a negated auxiliary."
    ),

    # 30. consensus vs concentration (Cognitive Malapropism / Contextual Word Choice)
    (
        r'\b((?:(?:completely|utterly|totally|severely|badly|deeply|easily)\s+)?(?:overwhelm\w*|disturb\w*|disrupt\w*|shatter\w*|ruin\w*|broke|broken|break|lose|lost|losing))\s+(?:(?:completely|utterly|totally|severely)\s+)?((?:their|his|her|my|our|one\'s|the)\s+)consensus\b',
        r'\1 \2concentration',
        "Contextual Word Choice (Malapropism): In this cognitive/study context, 'concentration' (mental focus) is required instead of 'consensus' (general agreement)."
    ),
    (
        r'\b(deep|intense|unbroken|undivided|great|immense|lapsed?|lapse\s+in)\s+consensus\b',
        r'\1 concentration',
        "Contextual Word Choice (Malapropism): 'concentration' (mental focus) should be used instead of 'consensus' (general agreement)."
    ),

    # 31. elicit vs illicit
    (
        r'\b(illicit)\s+(a\s+response|an\s+answer|information|feedback|sympathy|support|laughter|a\s+reaction|truth)\b',
        r'elicit \2',
        "Confused word: use verb 'elicit' (to draw out or evoke), not adjective 'illicit' (illegal)."
    ),
    (
        r'\b(elicit)\s+(drugs|trade|activities|affair|operations|substances|market)\b',
        r'illicit \2',
        "Confused word: use adjective 'illicit' (illegal/forbidden), not verb 'elicit' (to evoke)."
    ),

    # 32. allusion vs illusion
    (
        r'\boptical\s+allusion\b',
        'optical illusion',
        "Use 'optical illusion' (visual deception), not 'allusion' (indirect reference)."
    ),
    (
        r'\bunder\s+the\s+allusion\b',
        'under the illusion',
        "Use 'under the illusion' (false belief), not 'allusion'."
    ),
    (
        r'\bmake\s+(?:an\s+)?illusion\s+to\b',
        'make an allusion to',
        "Use 'allusion to' (indirect reference), not 'illusion' (deception)."
    ),

    # 33. compliment vs complement
    (
        r'\b(pay|paid|pays|paying)\s+(?:(me|him|her|us|them|someone)\s+)?a\s+complement\b',
        lambda m: m.group(0).replace('complement', 'compliment').replace('Complement', 'Compliment'),
        "Confused word: use 'compliment' (expression of praise), not 'complement' (addition that enhances)."
    ),
    (
        r'\bperfect\s+compliment\s+to\b',
        'perfect complement to',
        "Confused word: use 'complement' (something that pairs well or completes), not 'compliment' (praise)."
    ),

    # 34. discreet vs discrete
    (
        r'\b(separate|distinct|individual)\s+and\s+discreet\b',
        r'\1 and discrete',
        "Confused word: use 'discrete' (distinct/separate), not 'discreet' (careful/circumspect)."
    ),
    (
        r'\bdiscreet\s+(units|steps|categories|variables|data|components|stages|entities)\b',
        r'discrete \1',
        "Confused word: use 'discrete' (separate and distinct), not 'discreet' (unobtrusive)."
    ),

    # 35. flout vs flaunt
    (
        r'\bflaunt(?:ed|ing|s)?\s+(?:the\s+)?(rules|law|laws|regulations|authority|tradition|ban)\b',
        lambda m: m.group(0).replace('flaunt', 'flout').replace('Flaunt', 'Flout'),
        "Confused word: use 'flout' (to openly disregard or disobey rules), not 'flaunt' (to show off)."
    ),

    # 36. eminent vs imminent
    (
        r'\beminent\s+(danger|threat|disaster|collapse|arrival|peril|demise|risk)\b',
        r'imminent \1',
        "Confused word: use 'imminent' (about to occur), not 'eminent' (distinguished/prominent)."
    ),

    # 37. adverse vs averse
    (
        r'\baverse\s+(effects?|conditions|weather|reaction|impact|consequences|circumstances)\b',
        r'adverse \1',
        "Confused word: use 'adverse' (unfavorable/harmful), not 'averse' (reluctant/opposed)."
    ),
    (
        r'\badverse\s+to\s+(risk|change|taking)\b',
        r'averse to \1',
        "Confused word: use 'averse to' (having a strong dislike or reluctance), not 'adverse'."
    ),

    # 38. perspective vs prospective
    (
        r'\bperspective\s+(students|clients|buyers|employers|candidates|customers|employees|investors)\b',
        r'prospective \1',
        "Confused word: use 'prospective' (potential or expected future), not 'perspective' (point of view)."
    ),
    (
        r'\bfrom\s+(?:(my|his|her|our|their|a|the)\s+)?prospective\b',
        lambda m: m.group(0).replace('prospective', 'perspective').replace('Prospective', 'Perspective'),
        "Confused word: use 'perspective' (point of view), not 'prospective' (future)."
    ),

    # 39. assure vs ensure
    (
        r'\bassure\s+that\b',
        'ensure that',
        "Confused word: use 'ensure that' (to make certain that), not 'assure' (to convince a person)."
    )
]


class HomophoneEngine:
    """Disambiguates contextual homophones and confused words."""

    @classmethod
    def apply(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        revised = text
        detected = []

        for item in CONFUSED_WORD_RULES:
            pat = item[0]
            repl = item[1]
            expl = item[2]

            matches = list(re.finditer(pat, revised, re.IGNORECASE))
            if matches:
                for m in matches:
                    detected.append({
                        "matched": m.group(0),
                        "explanation": expl
                    })
                revised = re.sub(pat, repl, revised, flags=re.IGNORECASE)

        return revised, detected
