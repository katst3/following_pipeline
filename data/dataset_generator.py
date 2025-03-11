import pickle
import json
import os
import random
import argparse
from tqdm import tqdm
import glob
from collections import defaultdict

DEFAULT_SEED = 42

# names list for actors
NAMES = [
    "Bob", "Alice", "Daniel", "Dorothy", "Paul", "Helen", "Jason", "Ruth", 
    "Michael", "Linda", "Brian", "Donna", "Matthew", "Betty", "Charles", 
    "Patricia", "James", "Susan", "George", "Sarah", "Richard", "Karen", 
    "Christopher", "Nancy", "Steven", "Carol", "Kevin", "Anna", "Edward", "Lisa"
]

# map densities to category names for tracking
DENSITY_CATEGORIES = {
    0.26: "simple",
    0.48: "deeper",
    0.50: "less_dense",
    0.58: "dense",
    0.68: "superdense"
}

# default actor ranges for each split
DEFAULT_ACTOR_RANGES = {
    'train': (2, 8),
    'valid': (2, 8),
    'validB': (9, 20),
    'test': (21, 30)
}

class Actor:
    def __init__(self, name, direction=None, n_directions=4):
        self.name = name
        self.direction_choices = {
            2: ["north", "south"],
            4: ["north", "east", "south", "west"],
        }
        if direction is not None:
            self.direction = direction
        else:
            self.direction = random.choice(self.direction_choices[n_directions])

        self.turn_dict = {
            "right": {
                "north": "east",
                "east": "south",
                "south": "west",
                "west": "north",
            },
            "around": {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
            },
            "left": {
                "north": "west",
                "west": "south",
                "south": "east",
                "east": "north",
            },
        }

    def follows(self, actor):
        self.direction = actor.direction
        return f"{self.name} follows {actor.name}. "

    def opposite_direction_of(self, actor):
        self.direction = self.turn_dict["around"][actor.direction]
        return f"{self.name} goes in the opposite direction of {actor.name}. "

    def turns(self, turn_direction):
        self.direction = self.turn_dict[turn_direction][self.direction]
        return f"{self.name} turns {turn_direction}. "


class Story:
    def __init__(self, actors, n_sentences, n_directions=4):
        self.actor_init = actors
        self.num_actors = len(actors)
        self.n_sentences = n_sentences
        self.events = ["follows", "turns", "op_dir_of"]
        self.turn_direction_choices = {
            2: ["around"],
            4: ["left", "right", "around"],
        }
        self.active_actors = []
        self.story = []
        self.n_directions = n_directions
        self.category = None
        self.interpersonal_count = 0
        self.total_events = 0

    def init_actor(self, actor):
        """initialize an actor by adding them to the active actors and recording their initial direction."""
        self.active_actors.append(actor)
        self.story.append(f"{actor.name} walks {actor.direction}. ")

    def generate_with_target_density(self, target_density):
        """generate a story with a specific density of interpersonal events."""
        # find closest density category
        closest_density = min(DENSITY_CATEGORIES.keys(), key=lambda x: abs(x - target_density))
        self.category = DENSITY_CATEGORIES[closest_density]
        
        # initialize actors
        for act in self.actor_init:
            self.init_actor(act)
        
        # calculate number of remaining events to generate
        remaining_events = self.n_sentences - self.num_actors
        
        # calculate how many should be interpersonal vs single-actor
        n_interpersonal = int(remaining_events * target_density)
        n_single = remaining_events - n_interpersonal
        
        # generate events
        events = []
        
        # generate single-actor events (turns)
        for _ in range(n_single):
            act = random.choice(self.active_actors)
            turn_direction = random.choice(self.turn_direction_choices[self.n_directions])
            events.append(act.turns(turn_direction))
        
        # generate interpersonal events
        for _ in range(n_interpersonal):
            act1, act2 = random.sample(self.active_actors, 2)
            if random.random() < 0.5:
                events.append(act1.follows(act2))
            else:
                events.append(act1.opposite_direction_of(act2))
        
        # shuffle and add to story
        random.shuffle(events)
        for event in events:
            self.story.append(event)
            
        # update interpersonal and total event counts
        self.total_events = remaining_events
        self.interpersonal_count = n_interpersonal
        
        return self.story


def track_story(story):
    """track the final direction of each actor in a story."""
    around_dict = {"north": "south", "east": "west", "south": "north", "west": "east"}
    right_dict = {"north": "east", "east": "south", "south": "west", "west": "north"}
    left_dict = {"north": "west", "east": "north", "south": "east", "west": "south"}
    actors = {}

    for sent in story:
        sent = sent.replace(".", "")
        words = sent.split()

        if words[0] not in actors:
            actors[words[0]] = words[-1]

        elif "follows" in sent:
            actors[words[0]] = actors[words[-1]]

        elif "opposite" in sent:
            actors[words[0]] = around_dict[actors[words[-1]]]

        elif "around" in sent:
            actors[words[0]] = around_dict[actors[words[0]]]

        elif "right" in sent:
            actors[words[0]] = right_dict[actors[words[0]]]

        elif "left" in sent:
            actors[words[0]] = left_dict[actors[words[0]]]

    return actors


def generate_story(num_actors, n_sentences, n_directions, target_density):
    """generate a single story with specified parameters."""
    # ensure we have enough names
    if num_actors > len(NAMES):
        additional_names = [f"Person{i}" for i in range(len(NAMES), num_actors)]
        names = NAMES + additional_names
    else:
        names = NAMES.copy()
    
    # shuffle names to avoid predictable patterns
    random.shuffle(names)
    
    # create actors
    actors = [Actor(name=names[i], n_directions=n_directions) for i in range(num_actors)]
    
    # create and generate story
    story = Story(actors, n_sentences, n_directions=n_directions)
    s = story.generate_with_target_density(target_density)
    
    # sample two actors to check if they're going in the same direction
    act1, act2 = random.sample(story.active_actors, 2)
    
    # create metadata
    metadata = {
        "category": story.category,
        "num_actors": num_actors,
        "n_sentences": n_sentences,
    }
    
    return (s, (act1.name, act2.name, act1.direction == act2.direction), metadata)


def generate_balanced_dataset(
    n_stories, 
    actor_range, 
    n_directions, 
    target_densities, 
    balance_mode="actor_only"
):
    """generate a dataset with balanced distribution based on specified mode."""
    stories_dict = {}
    
    min_actors, max_actors = actor_range
    actor_counts = list(range(min_actors, max_actors + 1))
    num_actor_values = len(actor_counts)
    num_densities = len(target_densities)
    
    # calculate target distribution based on balance mode
    if balance_mode == "actor_only":
        # each actor count gets approximately equal number of stories
        base_stories_per_actor = n_stories // num_actor_values
        remainder = n_stories % num_actor_values
        
        actor_targets = {}
        for i, actor_count in enumerate(actor_counts):
            actor_targets[actor_count] = base_stories_per_actor
            if i < remainder:
                actor_targets[actor_count] += 1
                
        # equal pos/neg split for each actor count
        target_dist = {}
        for actor_count, total in actor_targets.items():
            target_dist[actor_count] = {
                "pos": total // 2,
                "neg": total - (total // 2)  # handle odd numbers correctly
            }
            
        # for tracking progress
        current_dist = {actor_count: {"pos": 0, "neg": 0} for actor_count in actor_counts}
        
    elif balance_mode == "density_only":
        # each density gets approximately equal number of stories
        base_stories_per_density = n_stories // num_densities
        remainder = n_stories % num_densities
        
        density_targets = {}
        for i, density in enumerate(target_densities):
            density_targets[density] = base_stories_per_density
            if i < remainder:
                density_targets[density] += 1
                
        # equal pos/neg split for each density
        target_dist = {}
        for density, total in density_targets.items():
            density_name = DENSITY_CATEGORIES.get(density, f"density_{int(density*100)}")
            target_dist[density_name] = {
                "pos": total // 2,
                "neg": total - (total // 2)
            }
            
        # for tracking progress
        current_dist = {
            DENSITY_CATEGORIES.get(d, f"density_{int(d*100)}"): {"pos": 0, "neg": 0} 
            for d in target_densities
        }
        
    elif balance_mode == "both":
        # balanced across both actor counts and densities
        combinations = num_actor_values * num_densities
        base_stories_per_combo = n_stories // combinations
        remainder = n_stories % combinations
        
        # distribute evenly across all combinations
        target_dist = defaultdict(lambda: defaultdict(dict))
        combo_list = []
        
        for actor_count in actor_counts:
            for density in target_densities:
                density_name = DENSITY_CATEGORIES.get(density, f"density_{int(density*100)}")
                combo_list.append((actor_count, density_name))
        
        # distribute base amount
        for actor_count, density_name in combo_list:
            combo_total = base_stories_per_combo
            target_dist[actor_count][density_name] = {
                "pos": combo_total // 2,
                "neg": combo_total - (combo_total // 2)
            }
        
        # distribute remainder
        for i in range(remainder):
            actor_count, density_name = combo_list[i % len(combo_list)]
            pos_or_neg = "pos" if i % 2 == 0 else "neg"
            target_dist[actor_count][density_name][pos_or_neg] += 1
            
        # for tracking progress
        current_dist = defaultdict(lambda: defaultdict(lambda: {"pos": 0, "neg": 0}))
    
    else:
        raise ValueError(f"Invalid balance_mode: {balance_mode}")
    
    # progress bar
    pbar = tqdm(total=n_stories, desc=f"Generating stories (actor range: {actor_range})")
    
    # generate stories until we meet our targets
    attempts = 0
    max_attempts = n_stories * 20
    stories_generated = 0
    
    while stories_generated < n_stories and attempts < max_attempts:
        # determine what kind of story to generate next based on balance mode
        if balance_mode == "actor_only":
            # find actor counts that need more stories
            eligible_actors = []
            for actor_count in actor_counts:
                total_actor = current_dist[actor_count]["pos"] + current_dist[actor_count]["neg"]
                if total_actor < actor_targets[actor_count]:
                    eligible_actors.append(actor_count)
            
            if not eligible_actors:
                break  # all targets met
                
            # select an actor count
            num_actors = random.choice(eligible_actors)
            
            # determine if we need pos or neg to maintain balance
            pos_count = current_dist[num_actors]["pos"]
            neg_count = current_dist[num_actors]["neg"]
            pos_target = target_dist[num_actors]["pos"]
            neg_target = target_dist[num_actors]["neg"]
            
            if pos_count < pos_target and neg_count < neg_target:
                # both under target, slight preference to balance
                if abs(pos_count - neg_count) > 3:
                    if pos_count < neg_count:
                        target_outcome = "pos"
                    else:
                        target_outcome = "neg"
                else:
                    target_outcome = random.choice(["pos", "neg"])
            elif pos_count < pos_target:
                target_outcome = "pos"
            elif neg_count < neg_target:
                target_outcome = "neg"
            else:
                attempts += 1
                continue
                
            # randomly select target density
            target_density = random.choice(target_densities)
            
        elif balance_mode == "density_only":
            # find densities that need more stories
            eligible_densities = []
            for density in target_densities:
                density_name = DENSITY_CATEGORIES.get(density, f"density_{int(density*100)}")
                total_density = current_dist[density_name]["pos"] + current_dist[density_name]["neg"]
                if total_density < density_targets[density]:
                    eligible_densities.append((density, density_name))
            
            if not eligible_densities:
                break  # all targets met
                
            # select a density
            target_density, density_name = random.choice(eligible_densities)
            
            # determine if we need pos or neg to maintain balance
            pos_count = current_dist[density_name]["pos"]
            neg_count = current_dist[density_name]["neg"]
            pos_target = target_dist[density_name]["pos"]
            neg_target = target_dist[density_name]["neg"]
            
            if pos_count < pos_target and neg_count < neg_target:
                # both under target, slight preference to balance
                if abs(pos_count - neg_count) > 3:
                    if pos_count < neg_count:
                        target_outcome = "pos"
                    else:
                        target_outcome = "neg"
                else:
                    target_outcome = random.choice(["pos", "neg"])
            elif pos_count < pos_target:
                target_outcome = "pos"
            elif neg_count < neg_target:
                target_outcome = "neg"
            else:
                attempts += 1
                continue
                
            # randomly select actor count
            num_actors = random.randint(min_actors, max_actors)
            
        elif balance_mode == "both":
            # find combinations that need more stories
            eligible_combinations = []
            for actor_count in actor_counts:
                for density in target_densities:
                    density_name = DENSITY_CATEGORIES.get(density, f"density_{int(density*100)}")
                    pos_count = current_dist[actor_count][density_name]["pos"]
                    neg_count = current_dist[actor_count][density_name]["neg"]
                    pos_target = target_dist[actor_count][density_name]["pos"]
                    neg_target = target_dist[actor_count][density_name]["neg"]
                    
                    if pos_count < pos_target:
                        eligible_combinations.append((actor_count, density, density_name, "pos"))
                    if neg_count < neg_target:
                        eligible_combinations.append((actor_count, density, density_name, "neg"))
            
            if not eligible_combinations:
                break  # all targets met
                
            # select a combination
            num_actors, target_density, density_name, target_outcome = random.choice(eligible_combinations)
        
        # calculate appropriate sentence count
        min_sentences = max(num_actors + 2, 5)
        max_sentences = min(num_actors * 3, 90)
        if max_sentences < min_sentences:
            max_sentences = min_sentences + 5
            
        n_sentences = random.randint(min_sentences, max_sentences)
        
        # create dictionary key
        key = (num_actors, n_sentences)
        if key not in stories_dict:
            stories_dict[key] = {"pos": [], "neg": []}
        
        # generate the story
        story_tuple = generate_story(num_actors, n_sentences, n_directions, target_density)
        
        # check consistency
        story_sentences, direction_check, metadata = story_tuple
        actors = track_story(story_sentences)
        expected = direction_check[2]
        name1, name2 = direction_check[0], direction_check[1]
        
        # skip if actors aren't found
        if name1 not in actors or name2 not in actors:
            attempts += 1
            continue
            
        actual = actors[name1] == actors[name2]
        
        # skip inconsistent stories
        if actual != expected:
            attempts += 1
            continue
        
        # for density balancing modes, check if story category matches expected
        if balance_mode in ["density_only", "both"]:
            story_category = metadata.get("category", "unknown")
            if story_category != density_name:
                attempts += 1
                continue
        
        # add story if it matches target outcome (pos/neg)
        if actual and target_outcome == "pos":
            stories_dict[key]["pos"].append(story_tuple)
            if balance_mode == "actor_only":
                current_dist[num_actors]["pos"] += 1
            elif balance_mode == "density_only":
                current_dist[metadata["category"]]["pos"] += 1
            elif balance_mode == "both":
                current_dist[num_actors][metadata["category"]]["pos"] += 1
            stories_generated += 1
            pbar.update(1)
        elif not actual and target_outcome == "neg":
            stories_dict[key]["neg"].append(story_tuple)
            if balance_mode == "actor_only":
                current_dist[num_actors]["neg"] += 1
            elif balance_mode == "density_only":
                current_dist[metadata["category"]]["neg"] += 1
            elif balance_mode == "both":
                current_dist[num_actors][metadata["category"]]["neg"] += 1
            stories_generated += 1
            pbar.update(1)
        
        attempts += 1
    
    pbar.close()
    
    if attempts >= max_attempts:
        print(f"Warning: Reached maximum attempts ({max_attempts}), generated {stories_generated}/{n_stories} stories.")
    
    # calculate total pos/neg counts
    total_pos = 0
    total_neg = 0
    
    if balance_mode == "actor_only":
        for actor_count in actor_counts:
            total_pos += current_dist[actor_count]["pos"]
            total_neg += current_dist[actor_count]["neg"]
            
        # print actor distribution
        print("\nActor distribution:")
        for actor_count in sorted(actor_counts):
            pos = current_dist[actor_count]["pos"]
            neg = current_dist[actor_count]["neg"]
            total = pos + neg
            print(f"  - {actor_count} actors: {total} stories ({pos} pos, {neg} neg)")
            
    elif balance_mode == "density_only":
        for density_name in current_dist:
            total_pos += current_dist[density_name]["pos"]
            total_neg += current_dist[density_name]["neg"]
            
        # print density distribution
        print("\nDensity category distribution:")
        for density_name in sorted(current_dist.keys()):
            pos = current_dist[density_name]["pos"]
            neg = current_dist[density_name]["neg"]
            total = pos + neg
            print(f"  - {density_name}: {total} stories ({pos} pos, {neg} neg)")
            
    elif balance_mode == "both":
        # print detailed distribution by actor and density
        print("\nDetailed distribution:")
        for actor_count in sorted(actor_counts):
            actor_pos = 0
            actor_neg = 0
            actor_total = 0
            
            print(f"Actor count: {actor_count}")
            for density in sorted(target_densities, key=lambda d: DENSITY_CATEGORIES.get(d, "")):
                density_name = DENSITY_CATEGORIES.get(density, f"density_{int(density*100)}")
                pos = current_dist[actor_count][density_name]["pos"]
                neg = current_dist[actor_count][density_name]["neg"]
                combo_total = pos + neg
                
                actor_pos += pos
                actor_neg += neg
                actor_total += combo_total
                
                total_pos += pos
                total_neg += neg
                
                print(f"  - {density_name}: {combo_total} stories ({pos} pos, {neg} neg)")
                
            print(f"  Total for {actor_count} actors: {actor_total} stories ({actor_pos} pos, {actor_neg} neg)")
    
    print(f"\nTotal: {stories_generated} stories ({total_pos} pos, {total_neg} neg)")
    
    return stories_dict, total_pos, total_neg


def create_indices(all_stories):
    """create indices for different splits of the dataset."""
    train_indices = []
    valid_indices = []
    validB_indices = []
    test_indices = []
    
    index = 0
    
    # flatten the dictionary structure to create indices
    for key, group in all_stories.items():
        split = key[0] if isinstance(key, tuple) and len(key) > 0 else None
        
        # add positive stories
        for _ in group["pos"]:
            if split == 'train':
                train_indices.append(index)
            elif split == 'valid':
                valid_indices.append(index)
            elif split == 'validB':
                validB_indices.append(index)
            elif split == 'test':
                test_indices.append(index)
            index += 1
        
        # add negative stories
        for _ in group["neg"]:
            if split == 'train':
                train_indices.append(index)
            elif split == 'valid':
                valid_indices.append(index)
            elif split == 'validB':
                validB_indices.append(index)
            elif split == 'test':
                test_indices.append(index)
            index += 1
    
    return train_indices, valid_indices, validB_indices, test_indices


def create_metadata(all_stories):
    """create metadata for all stories."""
    metadata = {}
    index = 0
    
    for key, group in all_stories.items():
        # add metadata for positive stories
        for story in group["pos"]:
            if len(story) > 2 and isinstance(story[2], dict):
                metadata[str(index)] = story[2]
            index += 1
        
        # add negative stories
        for story in group["neg"]:
            if len(story) > 2 and isinstance(story[2], dict):
                metadata[str(index)] = story[2]
            index += 1
    
    return metadata


def analyze_dataset(all_stories):
    """print detailed statistics about the generated dataset."""
    print("\nDataset Statistics:")
    
    for split_name, stories_dict in all_stories.items():
        # count total stories in this split
        total_stories = 0
        pos_count = 0
        neg_count = 0
        actor_counts = []
        sentence_counts = []
        categories = {}
        
        for key, group in stories_dict.items():
            pos_stories = group["pos"]
            neg_stories = group["neg"]
            
            total_stories += len(pos_stories) + len(neg_stories)
            pos_count += len(pos_stories)
            neg_count += len(neg_stories)
            
            # extract metadata from stories
            for story in pos_stories + neg_stories:
                metadata = story[2]
                actor_counts.append(metadata["num_actors"])
                sentence_counts.append(metadata["n_sentences"])
                
                category = metadata["category"]
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
        
        print(f"\n{split_name} split ({total_stories} stories):")
        print(f"  - Positive examples: {pos_count} ({pos_count/max(1, total_stories)*100:.1f}%)")
        print(f"  - Negative examples: {neg_count} ({neg_count/max(1, total_stories)*100:.1f}%)")
        
        if actor_counts:
            print(f"  - Actor count: min={min(actor_counts)}, max={max(actor_counts)}, avg={sum(actor_counts)/len(actor_counts):.1f}")
        
        if sentence_counts:
            print(f"  - Sentence count: min={min(sentence_counts)}, max={max(sentence_counts)}, avg={sum(sentence_counts)/len(sentence_counts):.1f}")
        
        print(f"  - Category distribution:")
        for category, count in sorted(categories.items()):
            print(f"    * {category}: {count} stories ({count/max(1, total_stories)*100:.1f}%)")


def generate_complete_dataset(
    train_size, 
    valid_size, 
    validB_size, 
    test_size, 
    n_directions, 
    target_densities=None,
    balance_mode="actor_only",
    output_dir="./datasets_new",
    output_name=None,
    seed=42
):
    """generate a complete dataset with all splits."""
    # set random seed for reproducibility
    random.seed(seed)
    
    # default to all densities if not specified
    if target_densities is None:
        target_densities = list(DENSITY_CATEGORIES.keys())
    # convert single value to list if needed
    elif not isinstance(target_densities, list):
        target_densities = [target_densities]
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # generate splits
    print(f"Generating train split (actor range: {DEFAULT_ACTOR_RANGES['train']}, size: {train_size})")
    train_stories, train_pos, train_neg = generate_balanced_dataset(
        train_size, 
        DEFAULT_ACTOR_RANGES['train'], 
        n_directions, 
        target_densities,
        balance_mode
    )
    
    print(f"\nGenerating valid split (actor range: {DEFAULT_ACTOR_RANGES['valid']}, size: {valid_size})")
    valid_stories, valid_pos, valid_neg = generate_balanced_dataset(
        valid_size, 
        DEFAULT_ACTOR_RANGES['valid'], 
        n_directions, 
        target_densities,
        balance_mode
    )
    
    print(f"\nGenerating validB split (actor range: {DEFAULT_ACTOR_RANGES['validB']}, size: {validB_size})")
    validB_stories, validB_pos, validB_neg = generate_balanced_dataset(
        validB_size, 
        DEFAULT_ACTOR_RANGES['validB'], 
        n_directions, 
        target_densities,
        balance_mode
    )
    
    print(f"\nGenerating test split (actor range: {DEFAULT_ACTOR_RANGES['test']}, size: {test_size})")
    test_stories, test_pos, test_neg = generate_balanced_dataset(
        test_size, 
        DEFAULT_ACTOR_RANGES['test'], 
        n_directions, 
        target_densities,
        balance_mode
    )
    
    # create a combined dictionary for all stories
    all_stories = {}
    
    # add stories from each split, ensuring unique keys
    for key, group in train_stories.items():
        all_stories[('train',) + key] = group
    
    for key, group in valid_stories.items():
        all_stories[('valid',) + key] = group
    
    for key, group in validB_stories.items():
        all_stories[('validB',) + key] = group
    
    for key, group in test_stories.items():
        all_stories[('test',) + key] = group
    
    # organize splits separately for analysis
    all_splits = {
        'train': train_stories,
        'valid': valid_stories,
        'validB': validB_stories,
        'test': test_stories
    }
    
    # analyze the complete dataset
    analyze_dataset(all_splits)
    
    # create dataset name if not provided
    if output_name is None:
        balance_suffix = {
            "actor_only": "actor_balanced",
            "density_only": "density_balanced",
            "both": "fully_balanced"
        }.get(balance_mode, "balanced")
        
        # if using a single density, include it in the name
        if len(target_densities) == 1:
            density_name = DENSITY_CATEGORIES.get(target_densities[0], f"density_{int(target_densities[0]*100)}")
            dataset_name = f"{density_name}_{n_directions}dir_{balance_suffix}"
        else:
            dataset_name = f"direction_following_{n_directions}dir_{balance_suffix}"
    else:
        dataset_name = output_name
    
    # save the combined dataset
    dataset_path = os.path.join(output_dir, f'{dataset_name}.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump(all_stories, f)
    
    # create and save indices
    train_indices, valid_indices, validB_indices, test_indices = create_indices(all_stories)
    
    # save indices as JSON files
    train_indices_path = os.path.join(output_dir, f'train_indices_{dataset_name}.json')
    valid_indices_path = os.path.join(output_dir, f'valid_indices_{dataset_name}.json')
    validB_indices_path = os.path.join(output_dir, f'validB_indices_{dataset_name}.json')
    test_indices_path = os.path.join(output_dir, f'test_indices_{dataset_name}.json')
    
    with open(train_indices_path, 'w') as f:
        json.dump(train_indices, f)
    
    with open(valid_indices_path, 'w') as f:
        json.dump(valid_indices, f)
    
    with open(validB_indices_path, 'w') as f:
        json.dump(validB_indices, f)
    
    with open(test_indices_path, 'w') as f:
        json.dump(test_indices, f)
    
    # save metadata for all stories
    metadata = create_metadata(all_stories)
    metadata_path = os.path.join(output_dir, f'metadata_{dataset_name}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"\nDataset successfully saved to {dataset_path}")
    print(f"Splits: train={len(train_indices)}, valid={len(valid_indices)}, " + 
          f"validB={len(validB_indices)}, test={len(test_indices)}")
    
    return dataset_path


def generate_experiment_datasets(output_dir="./datasets_new", seed=DEFAULT_SEED):
    """generate all datasets from table 2 with balanced actor distribution."""
    # set random seed for reproducibility
    random.seed(seed)
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # define all dataset configurations from table 2
    datasets = {
        "2dir": {
            "simple": {
                "train": 392,     # ~80% of 492
                "valid": 100,     # ~20% of 492
                "validB": 864,
                "test": 480
            },
            "deeper": {
                "train": 0,
                "valid": 0,
                "validB": 750,
                "test": 500
            },
            "less_dense": {
                "train": 0,
                "valid": 0,
                "validB": 750,
                "test": 500
            },
            "dense": {
                "train": 0,
                "valid": 0,
                "validB": 750,
                "test": 500
            },
            "superdense": {
                "train": 0,
                "valid": 0,
                "validB": 750,
                "test": 500
            }
        },
        "4dir": {
            "simple": {
                "train": 392,     # ~80% of 492
                "valid": 100,     # ~20% of 492
                "validB": 864,
                "test": 480
            },
            "deeper": {
                "train": 150,
                "valid": 30,      # 20% of train
                "validB": 600,
                "test": 500
            },
            "less_dense": {
                "train": 150,
                "valid": 30,      # 20% of train
                "validB": 600,
                "test": 500
            },
            "dense": {
                "train": 150,
                "valid": 30,      # 20% of train
                "validB": 600,
                "test": 500
            },
            "superdense": {
                "train": 150,
                "valid": 30,      # 20% of train
                "validB": 600,
                "test": 500
            }
        }
    }

    # reverse lookup from category name to density value
    name_to_density = {v: k for k, v in DENSITY_CATEGORIES.items()}
    
    # track generated datasets
    generated_datasets = []
    
    # generate each dataset
    for dir_type, densities in datasets.items():
        n_dir = int(dir_type.replace("dir", ""))
        
        for density_name, splits in densities.items():
            # skip if no stories specified
            if all(count == 0 for count in splits.values()):
                print(f"\nSkipping {density_name}_{dir_type} (no stories specified)")
                continue
                
            density_value = name_to_density.get(density_name)
            if density_value is None:
                print(f"Warning: Unknown density category: {density_name}")
                continue
                
            # generate dataset with balanced actor distribution
            print(f"\n{'='*80}")
            print(f"Generating {density_name}_{dir_type} dataset")
            print(f"{'='*80}")
            
            dataset_path = generate_complete_dataset(
                train_size=splits["train"],
                valid_size=splits["valid"],
                validB_size=splits["validB"],
                test_size=splits["test"],
                n_directions=n_dir,
                target_densities=[density_value],
                balance_mode="actor_only",
                output_dir=output_dir,
                output_name=f"{density_name}_{dir_type}",
                seed=hash(f"{density_name}_{dir_type}") % 10000  # different seed for each dataset
            )
            
            generated_datasets.append(dataset_path)
    
    print(f"\nSuccessfully generated {len(generated_datasets)} datasets according to table 2 specifications")
    
    return generated_datasets


def merge_datasets_by_direction(data_dir, direction_type):
    """combine all datasets with the same direction type."""
    print(f"Combining {direction_type} datasets...")
    
    # find all dataset pickle files for this direction
    pickle_pattern = os.path.join(data_dir, f"*_{direction_type}.pkl")
    dataset_files = glob.glob(pickle_pattern)
    
    # filter out any "all_" files to prevent processing previous combined datasets
    dataset_files = [f for f in dataset_files if not os.path.basename(f).startswith("all_")]
    
    if not dataset_files:
        print(f"No {direction_type} datasets found in {data_dir}")
        return None
    
    print(f"Found {len(dataset_files)} datasets: {[os.path.basename(f) for f in dataset_files]}")
    
    # initialize the combined dataset structure
    combined_stories = {}
    
    # standardize split mapping for consistency
    split_mappings = {
        'validA': 'valid',  # map any 'validA' to 'valid'
        'validB': 'validb',  # map any 'validB' to 'validb'
        'valid': 'valid',    # keep 'valid' as is
        'train': 'train',    # keep 'train' as is
        'validb': 'validb',  # keep 'validb' as is
        'test': 'test'       # keep 'test' as is
    }
    
    # we'll keep track of all stories separately by split
    split_stories = {
        'train': [],
        'valid': [],
        'validb': [],
        'test': []
    }
    
    # process each dataset
    for dataset_file in dataset_files:
        dataset_name = os.path.basename(dataset_file).replace('.pkl', '')
        density_type = dataset_name.split('_')[0]
        print(f"\nProcessing {dataset_name} (density type: '{density_type}')...")
        
        # load the dataset
        with open(dataset_file, 'rb') as f:
            stories = pickle.load(f)
        
        # process each story and add to the appropriate split
        for key, group in stories.items():
            # get the split from the key
            if isinstance(key, tuple) and len(key) > 0:
                original_split = key[0]
                # map to standard split name
                std_split = split_mappings.get(original_split, original_split.lower())
                # create a new key without the split for the combined dataset
                new_key = key[1:] if len(key) > 1 else key
            else:
                # if key doesn't contain split info, we can't properly process it
                print(f"  Warning: Key {key} does not contain split information, skipping")
                continue
            
            # process positive stories
            for story in group["pos"]:
                # ensure story has proper metadata
                metadata = {}
                if len(story) > 2 and isinstance(story[2], dict):
                    metadata = dict(story[2])
                
                # make sure category is set correctly
                metadata['category'] = density_type
                
                # create updated story with proper metadata
                updated_story = (story[0], story[1], metadata)
                
                # add to appropriate split
                split_stories[std_split].append((new_key, "pos", updated_story))
            
            # process negative stories
            for story in group["neg"]:
                # ensure story has proper metadata
                metadata = {}
                if len(story) > 2 and isinstance(story[2], dict):
                    metadata = dict(story[2])
                
                # make sure category is set correctly
                metadata['category'] = density_type
                
                # create updated story with proper metadata
                updated_story = (story[0], story[1], metadata)
                
                # add to appropriate split
                split_stories[std_split].append((new_key, "neg", updated_story))
    
    # now reorganize into the combined_stories structure
    next_global_index = 0
    split_indices = {split: [] for split in split_stories}
    
    for split, stories_list in split_stories.items():
        for story_info in stories_list:
            key, story_type, story = story_info
            
            # add to combined stories
            if key not in combined_stories:
                combined_stories[key] = {"pos": [], "neg": []}
            
            combined_stories[key][story_type].append(story)
            
            # store index for this split
            split_indices[split].append(next_global_index)
            next_global_index += 1
    
    # save the combined dataset
    combined_name = f"all_{direction_type}"
    combined_stories_path = os.path.join(data_dir, f"{combined_name}.pkl")
    with open(combined_stories_path, 'wb') as f:
        pickle.dump(combined_stories, f)
    
    # save the indices with standardized names
    for split in ['train', 'valid', 'validb', 'test']:
        indices_path = os.path.join(data_dir, f"{split}_indices_{combined_name}.json")
        with open(indices_path, 'w') as f:
            json.dump(split_indices[split], f)
        print(f"Saved {len(split_indices[split])} {split} indices to {indices_path}")
    
    print(f"\nSuccessfully combined datasets into {combined_name}")
    print(f"Combined dataset has {len(split_indices['train'])} train, " + 
          f"{len(split_indices['valid'])} valid, " + 
          f"{len(split_indices['validb'])} validb, and " + 
          f"{len(split_indices['test'])} test examples")
    
    # return configuration
    return {
        'stories_file': combined_stories_path,
        'train_indices_file': os.path.join(data_dir, f"train_indices_{combined_name}.json"),
        'valid_indices_file': os.path.join(data_dir, f"valid_indices_{combined_name}.json"),
        'validb_indices_file': os.path.join(data_dir, f"validb_indices_{combined_name}.json"),
        'test_indices_file': os.path.join(data_dir, f"test_indices_{combined_name}.json")
    }


def main():
    """main entry point when script is run directly."""
    parser = argparse.ArgumentParser(description='Generate direction-following datasets with balanced distributions')
    
    # command selection
    parser.add_argument('--command', type=str, 
                        choices=['single_dataset', 'experiment_datasets', 'merge_datasets'],
                        default='single_dataset',
                        help='Command to execute')
    
    # split size parameters (for single_dataset)
    parser.add_argument('--train', type=int, default=200, 
                        help='Number of stories for train split')
    parser.add_argument('--valid', type=int, default=40, 
                        help='Number of stories for valid split')
    parser.add_argument('--validB', type=int, default=40, 
                        help='Number of stories for validB split')
    parser.add_argument('--test', type=int, default=40, 
                        help='Number of stories for test split')
    
    # dataset parameters
    parser.add_argument('--n_directions', type=int, default=4, choices=[2, 4],
                        help='Number of directions (2 or 4)')
    parser.add_argument('--density', type=float, default=None,
                        help='Target density value (e.g., 0.26, 0.48, 0.50, 0.58, 0.68)')
    parser.add_argument('--balance_mode', type=str, 
                        choices=['actor_only', 'density_only', 'both'],
                        default='actor_only',
                        help='How to balance the dataset')
    
    # merge parameters
    parser.add_argument('--dir_type', type=str, default=None, choices=['2dir', '4dir'],
                        help='Direction type for merging datasets')
    
    # other parameters
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./datasets',
                        help='Directory to save generated datasets')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Custom name for the output dataset')
    
    args = parser.parse_args()
    
    if args.command == 'single_dataset':
        # generate a single balanced dataset
        generate_complete_dataset(
            train_size=args.train,
            valid_size=args.valid,
            validB_size=args.validB,
            test_size=args.test,
            n_directions=args.n_directions,
            target_densities=[args.density] if args.density is not None else None,
            balance_mode=args.balance_mode,
            output_dir=args.output_dir,
            output_name=args.output_name,
            seed=args.seed
        )
        
    elif args.command == 'experiment_datasets':
        # generate all datasets from table 2
        generate_experiment_datasets(
            output_dir=args.output_dir,
            seed=args.seed
        )
        
    elif args.command == 'merge_datasets':
        if args.dir_type is None:
            print("Error: --dir_type must be specified for merge_datasets command")
            return
            
        # merge datasets by direction
        merge_datasets_by_direction(
            data_dir=args.output_dir,
            direction_type=args.dir_type
        )
    
    else:
        print(f"Error: Unknown command: {args.command}")


if __name__ == "__main__":
    main()