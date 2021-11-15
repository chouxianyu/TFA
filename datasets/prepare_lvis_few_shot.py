"""
为每个class采样k个instance（如果某class不足k个instance，则采样该class的所有instance）
"""

import argparse
import json
import os
import random

## 所有class的name，共1230个
CAT_NAMES = ['acorn', 'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'apple_juice', 'applesauce', 'apricot', 'apron', 'aquarium', 'armband', 'armchair', 'armoire', 'armor', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'ax', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo', 'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball_hoop', 'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)', 'battery', 'beachball', 'bead', 'beaker', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card', 'biscuit_(bread)', 'pirate_flag', 'black_sheep', 'blackboard', 'blanket', 'blazer', 'blender', 'blimp', 'blinker', 'blueberry', 'boar', 'gameboard', 'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'book_bag', 'bookcase', 'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'bowling_pin', 'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'breechcloth', 'bridal_gown', 'briefcase', 'bristle_brush', 'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer', 'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn', 'corned_beef', 'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butcher_knife', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candelabrum', 'candle', 'candle_holder', 'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'cannon', 'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower', 'caviar', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue', 'champagne', 'chandelier', 'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard', 'chest_of_drawers_(furniture)', 'chicken_(animal)', 'chicken_wire', 'chickpea', 'Chihuahua', 'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent', 'clementine', 'clip', 'clipboard', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffee_filter', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil', 'coin', 'colander', 'coleslaw', 'coloring_material', 'combination_lock', 'pacifier', 'comic_book', 'computer_keyboard', 'concrete_mixer', 'cone', 'control', 'convertible_(automobile)', 'sofa_bed', 'cookie', 'cookie_jar', 'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset', 'romaine_lettuce', 'costume', 'cougar', 'coverall', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'cracker', 'crape', 'crate', 'crayon', 'cream_pitcher', 'credit_card', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crouton', 'crow', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupcake', 'hair_curler', 'curling_iron', 'curtain', 'cushion', 'custard', 'cutting_tool', 'cylinder', 'cymbal', 'dachshund', 'dagger', 'dartboard', 'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk', 'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dishwasher_detergent', 'diskette', 'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dollar', 'dolphin', 'domestic_ass', 'eye_mask', 'doorbell', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drinking_fountain', 'drone', 'dropper', 'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'Dutch_oven', 'eagle', 'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'fish', 'fish_(food)', 'fishbowl', 'fishing_boat', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'fruit_salad', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'surgical_gown', 'grape', 'grasshopper', 'grater', 'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle', 'grillroom', 'grinder_(tool)', 'grits', 'grizzly', 'grocery_bag', 'guacamole', 'guitar', 'gull', 'gun', 'hair_spray', 'hairbrush', 'hairnet', 'hairpin', 'ham', 'hamburger', 'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'hatch', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headset', 'headstall_(for_horses)', 'hearing_aid', 'heart', 'heater', 'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce', 'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear', 'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate', 'ice_tea', 'igniter', 'incense', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jean', 'jeep', 'jelly_bean', 'jersey', 'jet_plane', 'jewelry', 'joystick', 'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono', 'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knight_(chess_piece)', 'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather', 'legging_(clothing)', 'Lego', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lightning_rod', 'lime', 'limousine', 'linen_paper', 'lion', 'lip_balm', 'lipstick', 'liquor', 'lizard', 'Loafer_(type_of_shoe)', 'log', 'lollipop', 'lotion', 'speaker_(stero_equipment)', 'loveseat', 'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallet', 'mammoth', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone', 'milk', 'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motor_vehicle', 'motorboat', 'motorcycle', 'mound_(baseball)', 'mouse_(animal_rodent)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument', 'nailfile', 'nameplate', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newsstand', 'nightshirt', 'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'oregano', 'ostrich', 'ottoman', 'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle', 'padlock', 'paintbox', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose', 'papaya', 'paperclip', 'paper_plate', 'paper_towel', 'paperback_book', 'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parchment', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport', 'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper', 'pepper_mill', 'perfume', 'persimmon', 'baby', 'pet', 'petfood', 'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow', 'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat', 'plate', 'platter', 'playing_card', 'playpen', 'pliers', 'plow_(farm_equipment)', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'police_van', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)', 'portrait', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'red_cabbage', 'reflector', 'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map', 'robe', 'rocking_chair', 'roller_skate', 'Rollerblade', 'rolling_pin', 'root_beer', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin', 'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse', 'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'scrambled_eggs', 'scraper', 'scratcher', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane', 'seashell', 'seedling', 'serving_dish', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass', 'shoulder_bag', 'shovel', 'shower_head', 'shower_curtain', 'shredder_(for_paper)', 'sieve', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'sled', 'sleeping_bag', 'sling_(bandage)', 'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'soda_fountain', 'carbonated_water', 'sofa', 'softball', 'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon', 'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steamer_(kitchen_appliance)', 'steering_wheel', 'stencil', 'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup', 'stockings_(leg_wear)', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'sunscreen', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table', 'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'telephoto_lens', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'tequila', 'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray', 'tree_house', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban', 'turkey_(bird)', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn', 'vacuum_cleaner', 'valve', 'vase', 'vending_machine', 'vent', 'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture', 'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe', 'wasabi', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_filter', 'water_heater', 'water_jug', 'water_gun', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whiskey', 'whistle', 'wick', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'wing_chair', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yak', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini']  # fmt: skip

## novel classes的id(index)，共454个
novel_classes = [
    0, 6, 9, 13, 14, 15, 20, 21, 30, 37, 38, 39, 41, 45, 48, 50, 51, 63,
    64, 69, 71, 73, 82, 85, 93, 99, 100, 104, 105, 106, 112, 115, 116,
    119, 121, 124, 126, 129, 130, 135, 139, 141, 142, 143, 146, 149,
    154, 158, 160, 162, 163, 166, 168, 172, 180, 181, 183, 195, 198,
    202, 204, 205, 208, 212, 213, 216, 217, 218, 225, 226, 230, 235,
    237, 238, 240, 241, 242, 244, 245, 248, 249, 250, 251, 252, 254,
    257, 258, 264, 265, 269, 270, 272, 279, 283, 286, 290, 292, 294,
    295, 297, 299, 302, 303, 305, 306, 309, 310, 312, 315, 316, 317,
    319, 320, 321, 323, 325, 327, 328, 329, 334, 335, 341, 343, 349,
    350, 353, 355, 356, 357, 358, 359, 360, 365, 367, 368, 369, 371,
    377, 378, 384, 385, 387, 388, 392, 393, 401, 402, 403, 405, 407,
    410, 412, 413, 416, 419, 420, 422, 426, 429, 432, 433, 434, 437,
    438, 440, 441, 445, 453, 454, 455, 461, 463, 468, 472, 475, 476,
    477, 482, 484, 485, 487, 488, 492, 494, 495, 497, 508, 509, 511,
    513, 514, 515, 517, 520, 523, 524, 525, 526, 529, 533, 540, 541,
    542, 544, 547, 550, 551, 552, 554, 555, 561, 563, 568, 571, 572,
    580, 581, 583, 584, 585, 586, 589, 591, 592, 593, 595, 596, 599,
    601, 604, 608, 609, 611, 612, 615, 616, 625, 626, 628, 629, 630,
    633, 635, 642, 644, 645, 649, 655, 657, 658, 662, 663, 664, 670,
    673, 675, 676, 682, 683, 685, 689, 695, 697, 699, 702, 711, 712,
    715, 721, 722, 723, 724, 726, 729, 731, 733, 734, 738, 740, 741,
    744, 748, 754, 758, 764, 766, 767, 768, 771, 772, 774, 776, 777,
    781, 782, 784, 789, 790, 794, 795, 796, 798, 799, 803, 805, 806,
    807, 808, 815, 817, 820, 821, 822, 824, 825, 827, 832, 833, 835,
    836, 840, 842, 844, 846, 856, 862, 863, 864, 865, 866, 868, 869,
    870, 871, 872, 875, 877, 882, 886, 892, 893, 897, 898, 900, 901,
    904, 905, 907, 915, 918, 919, 920, 921, 922, 926, 927, 930, 931,
    933, 939, 940, 944, 945, 946, 948, 950, 951, 953, 954, 955, 956,
    958, 959, 961, 962, 963, 969, 974, 975, 988, 990, 991, 998, 999,
    1001, 1003, 1005, 1008, 1009, 1010, 1012, 1015, 1020, 1022, 1025,
    1026, 1028, 1029, 1032, 1033, 1046, 1047, 1048, 1049, 1050, 1055,
    1066, 1067, 1068, 1072, 1073, 1076, 1077, 1086, 1094, 1099, 1103,
    1111, 1132, 1135, 1137, 1138, 1139, 1140, 1144, 1146, 1148, 1150,
    1152, 1153, 1156, 1158, 1165, 1166, 1167, 1168, 1169, 1171, 1178,
    1179, 1180, 1186, 1187, 1188, 1189, 1203, 1204, 1205, 1213, 1215,
    1218, 1224, 1225, 1227
]  # fmt: skip

## base classes的id(index)，共776个
base_classes = [c for c in range(1230) if c not in novel_classes]


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据集标注（json）文件路径
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/lvis/lvis_v0.5_train.json",
        help="path to the annotation file",
    )
    # 需要有几个shot，为每个class采样k个instance
    parser.add_argument(
        "--shots", type=int, default=10, help="number of shots"
    )
    args = parser.parse_args()
    return args


def get_shots(args):
    ## 读取标注文件
    data = json.load(open(args.data, "r"))
    ann = data["annotations"] # list[dict]，693958个标注，每个标注是一个dict


    ## 获取每个class的所有标注
    anno_cat = {i: [] for i in range(1230)} # 每个cls的所有标注：key为cls_id，value为list[dict]，其中每个dict表示一个标注
    for a in ann: # 遍历每个标注（dict）
        anno_cat[a["category_id"] - 1].append(a)


    ## 为每个class采样k个instance（如果某class不足k个instance，则采样该class的所有instance）
    anno = [] # list[dict]，每个dict表示一个标注
    for i, c in enumerate(CAT_NAMES): # 遍历所有class
        if len(anno_cat[i]) < args.shots: # 如果该class不足k个instance，则采样该class的所有instance
            shots = anno_cat[i]
        else:
            shots = random.sample(anno_cat[i], args.shots) # 为该class随机采样k个instance
        anno.extend(shots) # 将采样到的instance记录下来


    ## 保存采样结果
    new_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
        "images": data["images"],
        "annotations": anno,
    }
    save_path = os.path.join("datasets/lvissplit", "lvis_shots.json")
    with open(save_path, "w") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    random.seed(421)

    args = parse_args()
    get_shots(args)
