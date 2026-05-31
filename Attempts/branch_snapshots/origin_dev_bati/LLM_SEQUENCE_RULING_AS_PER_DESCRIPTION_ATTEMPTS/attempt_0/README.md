# LLM Sequence Ruling From Descriptions - Attempt 0

Date: 2026-05-30

## Purpose

This attempt asks: can an LLM use the repo's step descriptions plus public semiconductor-process references to derive extra sequence-ordering rules that are not just copied from the existing validator?

The output is intentionally a candidate rule pack, not ground truth. A future solution can use it as:

- a reranker for next-step predictions,
- a constraint layer for completion,
- an anomaly explanation prior,
- a feature generator for step occurrence context.

## Inputs Used

Repo inputs:

- `training_data/*_Longdescr.csv`
- `training_data/*_longdescription_parameters.csv`
- `training_data/generation_rules.md`
- `training_data/*_variants.csv`
- existing solution code under `solutions/`

Key observation: existing solution loaders mostly use long-format `SEQUENCE_ID, STEP` traces. They skip the description/parameter files. That means the descriptions are still available as unused context.

## What The Description Files Contain

The description files provide step-level semantic hints, for example:

- `EPITAXIAL DEPOSITION` grows an epitaxial silicon layer and mentions temperature, gas flow, and deposition rate.
- `SPIN COAT PHOTORESIST` records spin speed, resist thickness, and coating quality.
- `EXPOSE LITHO LEVEL 1` mentions dose and focus.
- `OXIDE ETCH` mentions etch rate, selectivity, and endpoint control.
- parameter files add typical recipe hints such as RIE gases, RF power, implant dose/energy, furnace temperature, pressure, and thickness targets.

These are not per-lot measurements. They are generic process knowledge keyed by step type and family. Some step names repeat in different local contexts, so a future implementation should not blindly join on `STEP` alone.

## Online References Used

The public sources below were used to infer domain-ordering logic. They are not vendor-specific Infineon rules; they support general semiconductor manufacturing constraints.

- NIST wet chemical cleaning reference: cleaning is used before critical operations such as epitaxial growth, metal/dielectric film deposition, and diffusion.  
  https://www.nist.gov/publications/wet-chemical-cleaning-plasma-oxide-grown-heated-001-inp-surfaces
- UBC Nanofab BOE/HF dip page: HF/BOE is used to remove native silicon dioxide.  
  https://nanofab.ubc.ca/processes/buffer-oxide-etch-boe-dip
- UniversityWafer RCA cleaning overview: RCA clean and HF-last cleans are used around oxidation/deposition workflows.  
  https://www.universitywafer.com/rca-cleaning-process.html
- SemiconductorX photolithography overview: resist coat, bake, expose, develop, then pattern transfer through etch or implantation.  
  https://semiconductorx.com/mfg-front-end-photolithography.html
- TechTarget semiconductor process overview: deposition, lithography, etching, ion implantation, annealing, and related step purposes.  
  https://www.techtarget.com/whatis/feature/core-steps-in-the-semiconductor-manufacturing-process
- University of Virginia doping lecture: post-implant thermal processing repairs damage and electrically activates dopants.  
  https://www.virginia.edu/ms/research/wadley/high-temp/doping.htm
- NIST TSV copper electrodeposition reference: supports via-fill context for through-silicon vias.  
  https://www.nist.gov/publications/simulation-copper-electrodeposition-millimeter-size-through-silicon-vias
- OSHA semiconductor manufacturing chapter: wafer probing/sort happens after fabrication before assembly/packaging.  
  https://www.osha.gov/semiconductors
- Wevolver passivation overview: passivation protects semiconductor surfaces from moisture, contaminants, and electrical degradation.  
  https://www.wevolver.com/article/passivation-techniques-in-semiconductor-manufacturing

## Output Files

- `candidate_rules.json` - machine-readable candidate rule pack.
- `integration_notes.md` - how to use the rule pack in a future solution.

## Confidence

High confidence:

- lithography internal order,
- develop-before-patterned-etch,
- clean/surface-prep before deposition or oxidation,
- implant followed by anneal,
- CMP after deposition/fill,
- passivation before final test/ship,
- wafer sort before ship.

Medium confidence:

- exact step-window sizes,
- context-based disambiguation of repeated step names,
- how harshly to penalize missing measurements,
- using parameter hints as model features rather than hard rules.

Low confidence:

- anything pretending to be real Infineon production routing.
