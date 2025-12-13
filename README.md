# BioGraph, A Hi-C Visualization Tool

This project provides an interactive visualization pipeline for exploring Hi-C contact maps, Topologically Associating Domains (TADs), and community structures using graph-based methods. It supports loading .RAWobserved Hi-C matrices, detecting TAD clusters, identifying Louvain communities, and displaying these features dynamically on both a heatmap and graph layout.

The project can be find at: https://github.com/shinychikapu/CMPE195-Senior-Project

## Introduction

Hi-C data captures the three-dimensional organization of the genome by measuring contact frequencies between pairs of genomic regions. Traditionally, these interactions are visualized as heatmaps, which provide a matrix-level view of chromatin structure but can make it difficult to interpret higher-order spatial relationships. In this project, we aim to offer an alternative visualization approach by converting Hi-C contact maps into graph representations. This allows users to explore community structure, TAD organization, and long-range interactions in a more intuitive and interactive way than conventional heatmaps alone.

## Features
- Load and visualize Hi-C contact maps interactively
- Graph-based representation of genomic bins
- Louvain community detection clustering
- TAD/Loop overlay coloring
- Heatmap Integration
- Graph visulization configurations

## Usage
1. Clone the repository
2. Load the webpage locally (mgv.html). **VSCode Live Server** can be used for this. 
3. Upload your rawObserved Hi-C matrix, it should have three columns for i,j,v
<img width="365" height="58" alt="image" src="https://github.com/user-attachments/assets/14deeb4e-1912-4be9-91ba-01f781b3f245" />

4. Optinonally, upload your annotation file for TAD or Loop with appropriate headers for overlays. Turn overlay toggle on to replace Louvain clustering with TAD/Loop
<img width="367" height="31" alt="image" src="https://github.com/user-attachments/assets/abb6422d-98b5-494a-8b89-e5edc0eb55e3" />
<img width="373" height="29" alt="image" src="https://github.com/user-attachments/assets/40d36c4a-cfdd-4f38-aab6-9a23a047fcc0" />

5. Modify the configurations to your likings
<img width="380" height="551" alt="image" src="https://github.com/user-attachments/assets/df47654c-90ae-48a4-b994-954fa6e5dbc0" />

## Data Format
Files on main branch are provided for testing. They are taken from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525. 
This is research data for the paper "A three-dimensional map of the human genome at kilobase resolution reveals prinicples of chromatin looping". The data to be visulized should have appropriate headers as specificed below
1. .rawObserved
   
Three columns which are values for i, j, v

2. TAD Annotation
   
chr1	x1	x2	chr2	y1	y2	color	f1	f2	f3	f4	f5

3. Loop Annotation

chr1	x1	x2	chr2	y1	y2	color	o	e_bl	e_donut	e_h	e_v	fdr_bl	fdr_donut	fdr_h	fdr_v	num_collapsed	centroid1	centroid2	radius

**Please refer to these files as examples**
1. .rawObserved: chr1_1mb.RAWobserved
2. TAD: GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt
3. Loop: GSE63525_GM12878_primary+replicate_HiCCUPS_looplist.txt
