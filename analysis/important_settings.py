# Important file for BOAT's correct function

import numpy as np
import os
import time
import getpass

bird = """
       _.--.__                                             _.--.
    ./'       `--.__                                   ..-'   ,'
  ,/               |`-.__                            .'     ./
 :,                 :    `--_    __                .'   ,./'_.....
 :                  :   /    `-:' _\.            .'   ./..-'   _.'
 :                  ' ,'       : / \ :         .'    `-'__...-'
 `.               .'  .        : \@/ :       .'       '------.,
    ._....____  ./    :     .. `     :    .-'      _____.----'
              `------------' : |     `..-'        `---.
                         .---'  :    ./      _._-----'
.---------._____________ `-.__/ : /`      ./_-----/':
`---...--.              `-_|    `.`-._______-'  /  / ,-----.__----.
   ,----' ,__.  .          |   /  `\.________./  ====__....._____.'
   `-___--.-' ./. .-._-'----\.                  ./.---..____.--.
         :_.-' '-'            `..            .-'===.__________.'
                                 `--...__.--'
    """

banner = """
 ____   ___    _  _____ _____ _ _       _     _   ____  _           
| __ ) / _ \  / \|_   _|  ___| (_) __ _| |__ | |_/ ___|(_)_ __ ___  
|  _ \| | | |/ _ \ | | | |_  | | |/ _` | '_ \| __\___ \| | '_ ` _ \ 
| |_) | |_| / ___ \| | |  _| | | | (_| | | | | |_ ___) | | | | | | |
|____/ \___/_/   \_\_| |_|   |_|_|\__, |_| |_|\__|____/|_|_| |_| |_|
                                  |___/    
"""

high_score = """
   _   _ _       _       ____                     _ 
  | | | (_) __ _| |__   / ___|  ___ ___  _ __ ___| |
  | |_| | |/ _` | '_ \  \___ \ / __/ _ \| '__/ _ \ |
  |  _  | | (_| | | | |  ___) | (_| (_) | | |  __/_|
  |_| |_|_|\__, |_| |_| |____/ \___\___/|_|  \___(_)
           |___/                                    
"""

plane = """
 __
 \  \     _ _
  \**\ ___\/ \\
X*#####*+^^\_\\
  o/\  \\
     \__\\
"""

plane_2 = """
 __
 \  \     _ _
  \**\ ___\/ \\
+*#####*+^^\_\\
  o/\  \\
     \__\\
"""

def high_performance_flight_simulator():
    shapes = ['-', '/', '|', "\\"]
    file_sort = ['texture', 'world', 'colorscheme', 'model', 'physics engine', 'avionics', 'Navier-Stokes solver']
    file_type = ['configuration file', 'license', 'libraries', 'executables']
    planes = [plane, plane_2]

    os.system('clear')

    count = 0

    while count < 13:
        print(banner)
        print(15*" " + shapes[count % len(shapes)] + " Loading {} {}...".format(np.random.choice(file_sort), np.random.choice(file_type)))
        time.sleep(np.random.uniform(0.2, 0.7))
        os.system('clear')
        count += 1

    print(banner)
    print(15*" " + "Loading done!")

    time.sleep(1)

    distance = 45 

    while distance > 4:
        frame = planes[distance%2]
        os.system('clear')
        for line in frame.splitlines():
            print(distance*" " + line)
        time.sleep(0.15)
        distance -= 1

    time.sleep(0.85)
    os.system('clear')

    score = np.random.randint(100, 5000)

    try:
        with open("important_config.boat", "r") as f:
            current_hs = f.readline()
            current_hs_holder = f.readline()
    except IOError:
        current_hs = "2500"
        current_hs_holder = "River" 

    if not current_hs.rstrip().isnumeric():
        print("Corrupt score file. Resetting to previous champion.")
        current_hs = 2500
        current_hs_holder = "River"
    else:
        current_hs = int(current_hs)

    if score > current_hs:
        print(high_score)
        print(2*" " + "You've got the high score with {} points, "
              "and displaced {}!\n".format(score, current_hs_holder))

        with open("important_config.boat", "w") as f:
            f.write(str(score) + "\n")
            f.write(getpass.getuser())
    else:
        print(5*" " + "HIGH SCORE: {} - {}".format(current_hs_holder, current_hs))
        print(5*" " + "You've won with {} points! Try again?\n".format(score))
