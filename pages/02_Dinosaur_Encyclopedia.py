import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image

st.set_page_config(
    page_title="PaleoNet - Dinosaur Encyclopedia", page_icon="📖", layout="wide"
)

st.title("Dinosaur Encyclopedia")

st.markdown(
    """
This page provides information about each of the 15 dinosaur species that our model can classify.
Learn about their characteristics, time periods, and interesting facts!
"""
)

# Dinosaur information dictionary
dino_info = {
    "Ankylosaurus": {
        "period": "Late Cretaceous (68-66 million years ago)",
        "diet": "Herbivore",
        "length": "8 meters (26 feet)",
        "weight": "4-8 tons",
        "description": "Ankylosaurus was a heavily armored dinosaur with a club-like tail that it could swing as a weapon. Its back was covered with bony plates (osteoderms) and spikes for protection against predators. Despite its fearsome appearance, it was a peaceful plant-eater.",
        "interesting_fact": "The club on its tail could swing with enough force to break the bones of attacking predators like Tyrannosaurus Rex.",
    },
    "Brachiosaurus": {
        "period": "Late Jurassic (154-153 million years ago)",
        "diet": "Herbivore",
        "length": "25 meters (82 feet)",
        "weight": "30-50 tons",
        "description": "Brachiosaurus was one of the tallest dinosaurs, with a long neck that allowed it to feed on foliage high in trees. Unlike many other long-necked dinosaurs, its front legs were longer than its hind legs, giving it a distinctive upward-sloping posture.",
        "interesting_fact": "Its nostrils were located on top of its head, which led scientists to once believe it lived underwater like a hippo. This theory has since been disproven.",
    },
    "Compsognathus": {
        "period": "Late Jurassic (150-145 million years ago)",
        "diet": "Carnivore",
        "length": "1 meter (3.3 feet)",
        "weight": "3 kilograms (6.6 pounds)",
        "description": "Compsognathus was one of the smallest dinosaurs, about the size of a chicken. Despite its small size, it was a swift and agile predator that hunted small lizards and mammals. It had sharp teeth and claws for catching and eating prey.",
        "interesting_fact": "A fossil of Compsognathus was found with a small lizard in its stomach, giving us rare direct evidence of what it ate.",
    },
    "Corythosaurus": {
        "period": "Late Cretaceous (77-75 million years ago)",
        "diet": "Herbivore",
        "length": "9 meters (30 feet)",
        "weight": "3-4 tons",
        "description": "Corythosaurus had a distinctive hollow crest on its head that may have been used for vocalizations and display. It belonged to the 'duck-billed' dinosaur family (hadrosaurs) and likely lived in herds for protection.",
        "interesting_fact": "Its crest contained nasal passages that may have allowed it to make loud, trumpet-like calls to communicate with others in its herd.",
    },
    "Dilophosaurus": {
        "period": "Early Jurassic (193 million years ago)",
        "diet": "Carnivore",
        "length": "7 meters (23 feet)",
        "weight": "400 kilograms (880 pounds)",
        "description": "Dilophosaurus had two thin, bony crests on its head that were likely used for display or species recognition. It was portrayed in 'Jurassic Park' as having a neck frill and the ability to spit venom, but there is no evidence for either of these features in the fossil record.",
        "interesting_fact": "It had a notch in its upper jaw that gave it a weak bite, suggesting it may have been a scavenger or specialized in hunting smaller prey.",
    },
    "Dimorphodon": {
        "period": "Early Jurassic (175-200 million years ago)",
        "diet": "Carnivore (Fish and Insects)",
        "length": "1 meter (3.3 feet) wingspan",
        "weight": "1-2 kilograms (2.2-4.4 pounds)",
        "description": "Dimorphodon was a flying reptile (pterosaur) with a large head relative to its body. It had two types of teeth (hence the name 'di-morpho-don' meaning 'two-form-teeth'): large fangs at the front and smaller teeth behind them.",
        "interesting_fact": "Unlike modern birds, it couldn't fold its wings completely, suggesting it may have spent significant time on the ground, perhaps hunting like modern roadrunners.",
    },
    "Gallimimus": {
        "period": "Late Cretaceous (70 million years ago)",
        "diet": "Omnivore",
        "length": "6 meters (20 feet)",
        "weight": "400 kilograms (880 pounds)",
        "description": "Gallimimus was an ostrich-like dinosaur with a small head, long neck, and powerful legs. Its name means 'chicken mimic' due to its neck vertebrae resembling those of a chicken. It was one of the fastest dinosaurs and featured in the famous running scene in 'Jurassic Park'.",
        "interesting_fact": "It had a keratinous beak but no teeth, and may have used it to filter small animals and plants from water, similar to modern flamingos.",
    },
    "Microceratus": {
        "period": "Late Cretaceous (70-65 million years ago)",
        "diet": "Herbivore",
        "length": "0.6 meters (2 feet)",
        "weight": "3 kilograms (6.6 pounds)",
        "description": "Microceratus was a small ceratopsian (horned dinosaur) with a tiny frill and no horns. Despite being related to Triceratops, it was much smaller and more primitive. It likely lived in herds and used its beak to crop low-growing vegetation.",
        "interesting_fact": "It was one of the smallest known ceratopsians and may have been prey for many carnivorous dinosaurs and even large birds of the period.",
    },
    "Pachycephalosaurus": {
        "period": "Late Cretaceous (70-65 million years ago)",
        "diet": "Herbivore",
        "length": "4.5 meters (15 feet)",
        "weight": "450 kilograms (990 pounds)",
        "description": "Pachycephalosaurus had a thick, domed skull roof that could be up to 25 cm (10 inches) thick. Scientists believe males used these domes for head-butting contests to establish dominance, similar to modern bighorn sheep.",
        "interesting_fact": "Recent studies suggest that instead of direct head-butting, they may have been hitting each other's flanks, as direct dome-to-dome impacts might have caused brain damage.",
    },
    "Parasaurolophus": {
        "period": "Late Cretaceous (76-74 million years ago)",
        "diet": "Herbivore",
        "length": "10 meters (33 feet)",
        "weight": "2.5 tons",
        "description": "Parasaurolophus had a dramatic backward-curving hollow crest that extended from the back of its head. The crest contained elongated nasal passages that probably served as resonating chambers for making loud calls.",
        "interesting_fact": "Computer models suggest that it could produce low-frequency sounds similar to a trombone, with different species having different 'notes' based on the size and shape of their crests.",
    },
    "Spinosaurus": {
        "period": "Mid Cretaceous (99-93.5 million years ago)",
        "diet": "Carnivore (primarily fish)",
        "length": "15-18 meters (49-59 feet)",
        "weight": "7-20 tons",
        "description": "Spinosaurus had a sail-like structure on its back formed by elongated neural spines, which may have been used for display, temperature regulation, or fat storage. Recent discoveries suggest it had short legs and a paddle-like tail, indicating it was largely aquatic.",
        "interesting_fact": "It's the only known swimming dinosaur, with adaptations similar to modern crocodiles for hunting fish in rivers and lakes. It was larger than T. Rex, making it the largest known carnivorous dinosaur.",
    },
    "Stegosaurus": {
        "period": "Late Jurassic (155-150 million years ago)",
        "diet": "Herbivore",
        "length": "9 meters (30 feet)",
        "weight": "5-7 tons",
        "description": "Stegosaurus had distinctive upright plates along its back and spikes on its tail called a thagomizer. The plates may have been used for display or temperature regulation, while the tail spikes were definitely defensive weapons.",
        "interesting_fact": "It had a brain the size of a walnut (weighing around 80 grams), one of the smallest brain-to-body ratios of any dinosaur. This led to the myth that it had a 'second brain' in its hip region, which is not true.",
    },
    "Triceratops": {
        "period": "Late Cretaceous (68-66 million years ago)",
        "diet": "Herbivore",
        "length": "9 meters (30 feet)",
        "weight": "5-9 tons",
        "description": "Triceratops had three distinctive facial horns and a large frill extending from the back of its skull. The horns and frill were likely used for species recognition, courtship display, and defense against predators like Tyrannosaurus Rex.",
        "interesting_fact": "Its name means 'three-horned face,' and it's one of the last non-avian dinosaurs to exist before the mass extinction event. Over 50 skulls have been found, making it one of the best-documented dinosaurs.",
    },
    "Tyrannosaurus_Rex": {
        "period": "Late Cretaceous (68-66 million years ago)",
        "diet": "Carnivore",
        "length": "12-13 meters (40-43 feet)",
        "weight": "8-14 tons",
        "description": "Tyrannosaurus Rex was one of the largest land carnivores with powerful jaws containing banana-sized teeth. Despite popular culture often depicting it as a fast runner, studies suggest its top speed was likely 12-25 mph, and it had excellent vision and smell.",
        "interesting_fact": "Its arms were small but powerful, with two fingers and could lift up to 200 kg (440 pounds). A T. Rex named 'Sue' is the largest and most complete specimen ever found, with 90% of its bones recovered.",
    },
    "Velociraptor": {
        "period": "Late Cretaceous (75-71 million years ago)",
        "diet": "Carnivore",
        "length": "2 meters (6.8 feet)",
        "weight": "15-20 kilograms (33-44 pounds)",
        "description": "Velociraptor was much smaller than depicted in 'Jurassic Park' and had feathers like modern birds. It had a distinctive sickle-shaped claw on each foot that it likely used to slash at prey. It was a swift, intelligent predator that may have hunted in packs.",
        "interesting_fact": "A famous fossil shows a Velociraptor locked in combat with a Protoceratops, with the raptor's claw embedded in the dinosaur's neck and the Protoceratops biting the raptor's arm. Both died in this position, possibly buried by a sandstorm.",
    },
}

# Create a dropdown to select a dinosaur
selected_dino = st.selectbox("Select a dinosaur species:", list(dino_info.keys()))

# Create columns for the layout
col1, col2 = st.columns([1, 2])

with col1:
    # Display the dinosaur image
    test_dir = os.path.join("data", "dinosaur_dataset_split", "test")

    try:
        # Get a sample image for the selected dinosaur
        dino_dir = os.path.join(test_dir, selected_dino)
        if os.path.exists(dino_dir):
            image_files = [f for f in os.listdir(dino_dir) if f.endswith(".jpg")]
            if image_files:
                sample_image = os.path.join(dino_dir, image_files[0])
                st.image(sample_image, use_container_width=True)
            else:
                st.info(f"No sample images found for {selected_dino}")
        else:
            st.info(f"Directory not found for {selected_dino}")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        # Fallback placeholder
        st.info("Image not available")

    # Display basic info
    st.subheader("Basic Information")
    info = dino_info[selected_dino]
    st.markdown(f"**Time Period:** {info['period']}")
    st.markdown(f"**Diet:** {info['diet']}")
    st.markdown(f"**Length:** {info['length']}")
    st.markdown(f"**Weight:** {info['weight']}")

with col2:
    # Display the detailed description
    st.subheader("Description")
    st.write(info["description"])

    st.subheader("Interesting Fact")
    st.info(info["interesting_fact"])

    # Create a timeline visualization showing when this dinosaur lived
    st.subheader("Timeline")

    # Create a simple timeline visualization
    fig, ax = plt.subplots(figsize=(10, 3))

    # Create time periods with their ranges
    periods = {
        "Early Triassic": (251, 247),
        "Middle Triassic": (247, 237),
        "Late Triassic": (237, 201),
        "Early Jurassic": (201, 174),
        "Middle Jurassic": (174, 163),
        "Late Jurassic": (163, 145),
        "Early Cretaceous": (145, 100),
        "Late Cretaceous": (100, 66),
        "Extinction Event": (66, 65),
    }

    # Determine which period this dinosaur belongs to
    dino_period = selected_dino
    period_name = info["period"].split("(")[0].strip()

    # Extract the time range from the description (if available)
    time_range = None
    if "(" in info["period"] and "million years ago" in info["period"]:
        time_text = info["period"].split("(")[1].split(")")[0]
        if "-" in time_text:
            try:
                time_range = [float(x) for x in time_text.split("-")[0:2]]
                time_range = [x for x in time_range if not np.isnan(x)]
            except:
                pass
        elif "million years ago" in time_text:
            try:
                time_range = [float(time_text.split(" ")[0])]
            except:
                pass

    # Colors for the periods
    colors = {
        "Triassic": "#FF9999",
        "Jurassic": "#99FF99",
        "Cretaceous": "#9999FF",
        "Extinction Event": "#FF0000",
    }

    # Plot the timeline
    y_pos = 0
    for p_name, (start, end) in periods.items():
        # Determine the color based on the period name
        for period_key in colors:
            if period_key in p_name:
                color = colors[period_key]
                break
        else:
            color = "#EEEEEE"

        # Draw the period bar
        ax.barh(
            y_pos, start - end, left=end, height=0.5, color=color, edgecolor="black"
        )

        # Add the period name
        ax.text((start + end) / 2, y_pos, p_name, ha="center", va="center", fontsize=8)

    # Highlight this dinosaur's period if available
    if time_range:
        if len(time_range) == 2:
            min_time, max_time = time_range
        else:
            min_time = max_time = time_range[0]

        # Draw a marker for this dinosaur
        ax.barh(
            y_pos,
            min_time - max_time,
            left=max_time,
            height=0.5,
            color="gold",
            edgecolor="black",
            alpha=0.7,
        )
        ax.text(
            (min_time + max_time) / 2,
            y_pos,
            selected_dino,
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
        )

    # Set the plot properties
    ax.set_yticks([])
    ax.set_xlabel("Millions of Years Ago")
    ax.invert_xaxis()  # Invert x-axis to show older times on the left
    ax.set_title(f"Timeline: When {selected_dino} Lived")

    # Add extinction event marker
    ax.axvline(x=66, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.text(
        66,
        -0.25,
        "Extinction Event",
        ha="center",
        va="top",
        color="red",
        fontsize=8,
        rotation=90,
    )

    st.pyplot(fig)

# Dietary information section
st.subheader("Dietary Information")

# Create a pie chart for diet distribution
fig, ax = plt.subplots(figsize=(5, 5))

# Count the diets
diet_counts = {}
for dino, info in dino_info.items():
    diet = info["diet"].split("(")[0].strip()  # Remove parenthetical info
    if diet in diet_counts:
        diet_counts[diet] += 1
    else:
        diet_counts[diet] = 1

# Highlight the current dinosaur's diet
diet_colors = [f"#66{140 + (_i*30):0<2x}66" for _i in range(len(diet_counts))]
current_diet = dino_info[selected_dino]["diet"].split("(")[0].strip()
current_diet_idx = list(diet_counts.keys()).index(current_diet)
diet_colors[current_diet_idx] = "gold"

# Create the pie chart
pie_result = ax.pie(
    list(diet_counts.values()),
    labels=list(diet_counts.keys()),
    autopct="%1.1f%%",
    startangle=90,
    colors=diet_colors,
)

# Unpack the returned values based on their length
wedges, texts = pie_result[:2]
autotexts = pie_result[2] if len(pie_result) > 2 else []

# Emphasize the current diet in the chart
for i, autotext in enumerate(autotexts):
    if i == current_diet_idx:
        autotext.set_color("black")
        autotext.set_fontweight("bold")

plt.title(
    f"Diet Distribution Among All 15 Species\n({selected_dino}'s diet highlighted in gold)"
)
st.pyplot(fig)

# Additional information
st.markdown(
    """
---
### Evolutionary Relationships

Dinosaurs are classified into two major groups based on their hip structure:
- **Saurischia** (lizard-hipped): Includes theropods like T. Rex and Velociraptor, and sauropods like Brachiosaurus
- **Ornithischia** (bird-hipped): Includes ceratopsians like Triceratops, and ornithopods like Parasaurolophus

Despite the name, birds evolved from saurischian dinosaurs, specifically from small theropods similar to Velociraptor.

The classification of dinosaurs continues to evolve as new fossils are discovered and techniques for analysis improve.
"""
)

# Show related species
st.subheader("Related Species")

# Very simple relationship mapping (not scientifically accurate)
related_species = {
    "Ankylosaurus": ["Stegosaurus"],
    "Brachiosaurus": [],
    "Compsognathus": ["Velociraptor", "Gallimimus"],
    "Corythosaurus": ["Parasaurolophus"],
    "Dilophosaurus": ["Tyrannosaurus_Rex"],
    "Dimorphodon": [],
    "Gallimimus": ["Compsognathus", "Velociraptor"],
    "Microceratus": ["Triceratops", "Pachycephalosaurus"],
    "Pachycephalosaurus": ["Triceratops", "Microceratus"],
    "Parasaurolophus": ["Corythosaurus"],
    "Spinosaurus": ["Tyrannosaurus_Rex"],
    "Stegosaurus": ["Ankylosaurus"],
    "Triceratops": ["Microceratus", "Pachycephalosaurus"],
    "Tyrannosaurus_Rex": ["Dilophosaurus", "Spinosaurus", "Velociraptor"],
    "Velociraptor": ["Compsognathus", "Gallimimus", "Tyrannosaurus_Rex"],
}

if related_species[selected_dino]:
    related_cols = st.columns(min(3, len(related_species[selected_dino])))

    for i, related in enumerate(related_species[selected_dino]):
        col = related_cols[i % len(related_cols)]
        col.subheader(related.replace("_", " "))

        # Get an image for the related species
        try:
            related_dir = os.path.join(test_dir, related)
            if os.path.exists(related_dir):
                image_files = [f for f in os.listdir(related_dir) if f.endswith(".jpg")]
                if image_files:
                    sample_image = os.path.join(related_dir, image_files[0])
                    col.image(sample_image, use_container_width=True)
        except:
            col.warning("Image not available")

        sub_col1, sub_col2 = col.columns(2)
        sub_col1.metric("Period", dino_info[related]["period"].split("(")[0].strip())
        sub_col2.metric("Diet", dino_info[related]["diet"])
else:
    st.info(f"No closely related species in our database for {selected_dino}")

st.markdown(
    """
---
### References and Further Reading

- **Books:**
  - "The Complete Dinosaur" by M.K. Brett-Surman, Thomas R. Holtz Jr., and James O. Farlow
  - "Dinosaurs: The Most Complete, Up-to-Date Encyclopedia for Dinosaur Lovers of All Ages" by Thomas R. Holtz Jr.

- **Websites:**
  - [Natural History Museum](https://www.nhm.ac.uk/discover/dinosaurs.html)
  - [Smithsonian National Museum of Natural History](https://naturalhistory.si.edu/exhibits/dinosaurs)
  - [University of California Museum of Paleontology](https://ucmp.berkeley.edu/diapsids/dinosaur.html)

- **Academic Papers:**
  - "The Dinosauria, 2nd Edition" by David B. Weishampel, Peter Dodson, and Halszka Osmólska
  - Recent research in paleontology journals like "Journal of Vertebrate Paleontology" and "Paleobiology"
"""
)
