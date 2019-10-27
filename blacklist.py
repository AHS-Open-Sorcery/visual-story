# When your AI is too dirty for the judges

BLACKLIST = [
    'sex',
    'love',
    'fuck',
    'shit',
    'nigger',
    'intercourse'
]





















def blacklisted(output: str) -> bool:
    outlow = output.lower()
    for bad_word in BLACKLIST:
        if bad_word in outlow:
            return True
    return False
