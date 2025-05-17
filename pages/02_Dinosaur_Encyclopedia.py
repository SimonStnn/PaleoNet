import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils import DINO_INFO, display_dino_info

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

# Create a dropdown to select a dinosaur
selected_dino = st.selectbox("Select a dinosaur species:", list(DINO_INFO.keys()))

info = DINO_INFO[selected_dino]

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
    display_dino_info(selected_dino)

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
    time_range: list[float] | None = None
    if "(" in info["period"] and "million years ago" in info["period"]:
        time_text = info["period"].split("(")[1].split(")")[0]
        if "-" in time_text:
            time_range = [float(x) for x in time_text.split(" ", maxsplit=1)[0].split("-")[0:2]]
            time_range = [x for x in time_range if not np.isnan(x)]
        elif "million years ago" in time_text:
            time_range = [float(time_text.split(" ", maxsplit=1)[0])]

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
        color: str = "#EEEEEE"
        for period_key, period_color in colors.items():
            if period_key in p_name:
                color: str = period_color
                break

        # Draw the period bar
        ax.barh(
            y_pos, start - end, left=end, height=0.5, color=color, edgecolor="black"
        )

        # Add the period name
        ax.text(
            (start + end) / 2,
            y_pos,
            p_name,
            ha="center",
            va="center",
            fontsize=8,
            rotation=90,
        )

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
            rotation=90,
        )

    # Set the plot properties
    ax.set_yticks([])
    ax.set_xlabel("Millions of Years Ago")
    ax.invert_xaxis()  # Invert x-axis to show older times on the left
    ax.set_title(f"Timeline: When {selected_dino} Lived")

    # Add extinction event marker
    # ax.axvline(x=66, color="red", linestyle="--", linewidth=2, alpha=0.7)

    st.pyplot(fig)

# Dietary information section
st.subheader("Dietary Information")

# Create a pie chart for diet distribution
fig, ax = plt.subplots(figsize=(5, 5))

# Count the diets
diet_counts = {}
for dino, info in DINO_INFO.items():
    diet = info["diet"].split("(", maxsplit=1)[0].strip()  # Remove parenthetical info
    if diet in diet_counts:
        diet_counts[diet] += 1
    else:
        diet_counts[diet] = 1

# Highlight the current dinosaur's diet
diet_colors = [f"#66{140 + (_i*30):0<2x}66" for _i in range(len(diet_counts))]
current_diet = DINO_INFO[selected_dino]["diet"].split("(")[0].strip()
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
        sub_col1.metric("Period", DINO_INFO[related]["period"].split("(")[0].strip())
        sub_col2.metric("Diet", DINO_INFO[related]["diet"])
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
