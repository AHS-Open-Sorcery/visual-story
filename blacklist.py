# When your AI is too dirty for the judges

#trigger warning

#not for innocent eyes

#might be disturbing






















BLACKLIST = [
    'anal',
    'anus',
    'arse',
    'ass',
    'ballsack',
    'bastard',
    'bitch',
    'biatch',
    'bloody',
    'blowjob',
    'blow job',
    'bollock',
    'bollok',
    'boner',
    'boob',
    'bugger',
    'bum',
    'butt',
    'buttplug',
    'clitoris',
    'cock',
    'coon',
    'crap',
    'cunt',
    'damn',
    'dick',
    'dildo',
    'dyke',
    'fag',
    'feck',
    'fellate',
    'fellatio',
    'felching',
    'fuck',
    'f u c k',
    'fudgepacker',
    'fudge packer',
    'flange',
    'Goddamn',
    'God damn',
    'hell',
    'homo',
    'jerk',
    'jizz',
    'knobend',
    'knob end',
    'labia',
    'muff',
    'nigger',
    'nigga',
    'penis',
    'piss',
    'pube',
    'pussy',
    'queer',
    'scrotum',
    'shit',
    'slut',
    'tit',
    'turd',
    'twat',
    'wank',
    'whore'
]





















def blacklisted(output: str) -> bool:
    outlow = output.lower()
    for bad_word in BLACKLIST:
        if bad_word in outlow:
            return True
    return False
