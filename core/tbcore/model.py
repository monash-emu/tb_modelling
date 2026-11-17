"""Construction of the TB transmission model.

Build order matters and is enforced here:

1. Natural history on an unstratified model.
2. Detection, treatment and any screening programmes.
3. Age stratification — first, because it carries the mixing matrix.
4. Optional stratifications, in the order the configuration lists them.
5. Births and deaths — last, because death flows are age-specific in both source and
   destination, and births must be split across every stratum that exists by then.
6. Derived output requests.
"""

from itertools import product

import jax.numpy as jnp
import pandas as pd
from summer2 import CompartmentalModel
from summer2.functions import time as stf
from summer2.parameters import Function, Parameter, Time

from tbcore.compartments import (
    ACTIVE_COMPS,
    COMPARTMENTS,
    INFECTIOUS_COMPS,
    MTB_NAIVE,
    SUSCEPTIBLE_COMPS,
    is_asymptomatic,
)
from tbcore.config import ModelConfig
from tbcore.demography import get_death_rates_by_age, get_population_over_time
from tbcore.mixing import get_age_mixing_matrix
from tbcore.outputs import request_model_outputs
from tbcore.stratifications.age import stratify_by_age
from tbcore.stratifications.base import BuildContext, FlowRegistry

#: Year used to rescale the raw transmission rate, so that the parameter keeps the same
#: order of magnitude whichever transmission mode the population-size exponent selects.
TRANSMISSION_REFERENCE_YEAR = 2020


def get_tb_model(
    config: ModelConfig,
    tv_params: dict,
    screening_programs: list = None,
    lever=None,
) -> CompartmentalModel:
    """Build a fully specified TB model.

    Parameters
    ----------
    config
        Model configuration.
    tv_params
        Time-variant parameters, keyed by name, each a year-indexed series. Must include
        ``tx_success_pct``.
    screening_programs
        Active screening campaigns. Empty for a baseline run.
    lever
        Optional :class:`~tbcore.interventions.ProgrammaticLever` perturbing routine
        services.
    """
    screening_programs = screening_programs or []
    age_groups = list(config.age_groups)

    single_age_pop, grouped_pop = get_population_over_time(
        config.iso3, age_groups, scaling_factor=config.pop_scaling
    )
    total_pop = grouped_pop.sum(axis=1)
    death_rate_funcs = get_death_rates_by_age(
        config.iso3, age_groups, config.start_time, grouped_pop
    )

    tsr = _treatment_success_func(tv_params, lever)
    neg_tx_outcome_funcs = get_neg_tx_outcome_funcs(death_rate_funcs, tsr)
    mixing_matrix = get_age_mixing_matrix(config.iso3, age_groups, config.heterogeneous_mixing)

    flows = FlowRegistry()
    model = _build_natural_history(config, total_pop, flows, lever)
    _add_detection_and_treatment(model, tsr, screening_programs, flows, lever)

    stratify_by_age(model, age_groups, neg_tx_outcome_funcs, screening_programs, mixing_matrix)

    context = BuildContext(flows=flows, screening_programs=screening_programs)
    for stratification in config.stratifications:
        stratification.apply(model, context)

    _add_births_and_deaths(
        model, config, total_pop, death_rate_funcs, neg_tx_outcome_funcs, flows
    )

    request_model_outputs(model, config, screening_programs, flows)

    # Retained for the economics layer, which weights age-specific DALY values by the
    # within-group age distribution.
    model.single_age_pop = single_age_pop
    return model


# --------------------------------------------------------------------------------------
# Natural history
# --------------------------------------------------------------------------------------


def _build_natural_history(
    config: ModelConfig, total_pop: pd.Series, flows: FlowRegistry, lever
) -> CompartmentalModel:
    model = CompartmentalModel(
        times=(config.start_time, config.end_time),
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
    )
    model.set_initial_population(
        distribution={
            MTB_NAIVE: total_pop.iloc[0] - config.seed,
            "clin_inf": config.seed,
        }
    )

    _add_infection_flows(model, total_pop, flows, lever)
    _add_early_infection_flows(model)
    _add_active_disease_flows(model)
    return model


def _add_infection_flows(model, total_pop: pd.Series, flows: FlowRegistry, lever) -> None:
    """Infection and reinfection, with a calibrated density/frequency dependence.

    The population-size exponent interpolates between density-dependent transmission (0),
    where population growth within a fixed habitable area raises transmission potential, and
    frequency-dependent transmission (1). The rate is rescaled by the population at a
    reference year so the raw parameter keeps a comparable magnitude across that range.
    """
    pop_exponent = Parameter("infection_pop_scale")
    if TRANSMISSION_REFERENCE_YEAR not in total_pop.index:
        raise ValueError(
            f"Population data must cover {TRANSMISSION_REFERENCE_YEAR}, used to rescale the "
            f"transmission rate; available years are "
            f"{int(total_pop.index.min())}-{int(total_pop.index.max())}."
        )
    reference_pop = total_pop.loc[TRANSMISSION_REFERENCE_YEAR]
    transmission_rate = Parameter("raw_transmission_rate") * reference_pop**pop_exponent

    if lever is not None and lever.transmission_rel != 1.0:
        transmission_rate = transmission_rate * lever.multiplier_func(lever.transmission_rel)

    for source in SUSCEPTIBLE_COMPS:
        if source == MTB_NAIVE:
            rel_susceptibility = 1.0
        elif source == "recovered":
            # Post-recovery and post-clearance susceptibility share a parameter: both
            # represent partial protection from prior infection without sterilising immunity.
            rel_susceptibility = Parameter("rel_sus_cleared")
        else:
            rel_susceptibility = Parameter(f"rel_sus_{source}")

        name = f"infection_from_{source}"
        model.add_infection_generalised_flow(
            name=name,
            gen_infection_exp=pop_exponent,
            contact_rate=transmission_rate * rel_susceptibility,
            source=source,
            dest="incipient",
        )
        flows.infection.append(name)


def _add_early_infection_flows(model) -> None:
    """Incipient, contained and cleared infection states."""
    # Rates set to 1.0 here are placeholders replaced by age-specific values during age
    # stratification; summer2 requires the flow to exist before it can be adjusted.
    model.add_transition_flow("containment", 1.0, "incipient", "contained")
    model.add_transition_flow("clearance", Parameter("clearance_rate"), "contained", "cleared")
    # Loss of immune control returns individuals to incipient infection, with renewed
    # progression risk.
    model.add_transition_flow("breakdown", Parameter("breakdown_rate"), "contained", "incipient")

    # Progression always produces asymptomatic disease first; the split is by infectiousness.
    progression_prop_inf = Parameter("progression_prop_infectious")
    model.add_transition_flow(
        "progression_lowinf", 1.0 - progression_prop_inf, "incipient", "subclin_lowinf"
    )
    model.add_transition_flow(
        "progression_inf", progression_prop_inf, "incipient", "subclin_inf"
    )


def _add_active_disease_flows(model) -> None:
    """Bidirectional movement along the symptom and infectiousness dimensions.

    Each transition changes one dimension only, per the ICE-TB framework.
    """
    for infectiousness in ["inf", "lowinf"]:
        model.add_transition_flow(
            f"clinical_progression_{infectiousness}",
            Parameter("clinical_progression_rate"),
            f"subclin_{infectiousness}",
            f"clin_{infectiousness}",
        )
        model.add_transition_flow(
            f"clinical_regression_{infectiousness}",
            Parameter("clinical_regression_rate"),
            f"clin_{infectiousness}",
            f"subclin_{infectiousness}",
        )

    for symptom_status in ["clin", "subclin"]:
        model.add_transition_flow(
            f"infectiousness_gain_{symptom_status}",
            Parameter("infectiousness_gain_rate"),
            f"{symptom_status}_lowinf",
            f"{symptom_status}_inf",
        )
        model.add_transition_flow(
            f"infectiousness_loss_{symptom_status}",
            Parameter("infectiousness_loss_rate"),
            f"{symptom_status}_inf",
            f"{symptom_status}_lowinf",
        )

    # Only asymptomatic disease resolves spontaneously.
    for infectiousness in ["inf", "lowinf"]:
        model.add_transition_flow(
            f"self_recovery_{infectiousness}",
            Parameter("self_recovery_rate"),
            f"subclin_{infectiousness}",
            "recovered",
        )


# --------------------------------------------------------------------------------------
# Detection, treatment and screening
# --------------------------------------------------------------------------------------


def _add_detection_and_treatment(
    model, tsr, screening_programs: list, flows: FlowRegistry, lever
) -> None:
    detection_rate = Parameter("recent_detection_rate") * Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("passive_detection_shape"),
            Parameter("passive_detection_inflection"),
            Parameter("passive_detection_past_frac"),
            1.0,
        ],
    )
    if lever is not None and lever.detection_rate_rel != 1.0:
        detection_rate = detection_rate * lever.multiplier_func(lever.detection_rate_rel)

    for compartment in ACTIVE_COMPS:
        # Asymptomatic disease is detected by routine services only to the extent that the
        # relative detection parameter allows; it is zero in the published configuration.
        multiplier = Parameter("rel_detection_subclin") if is_asymptomatic(compartment) else 1.0
        name = f"tb_detection_{compartment}"
        model.add_transition_flow(name, multiplier * detection_rate, compartment, "treatment")
        flows.passive_detection.append(name)

    model.add_computed_value_func("passive_detection_rate_clin", detection_rate)
    model.add_computed_value_func(
        "passive_detection_rate_subclin", Parameter("rel_detection_subclin") * detection_rate
    )

    for program in screening_programs:
        # Registered so that resource-use outputs can reconstruct the number of people
        # screened, which is a throughput times a population rather than a flow. Note this
        # is the throughput, not the detection hazard — see ScreeningProgram.
        model.add_computed_value_func(
            f"screening_throughput_{program.name}", program.throughput_func
        )
        for source_comp, sensitivity in program.scr_tool.sensitivities.items():
            name = f"{program.name}_{source_comp}"
            model.add_transition_flow(
                name,
                sensitivity * program.scr_tool.success_prop * program.raw_screening_func,
                source_comp,
                program.scr_tool.dest_comp,
            )
            flows.screening.append(name)

    model.add_transition_flow(
        "tx_recovery", tsr / Parameter("tx_duration"), "treatment", "recovered"
    )
    # Placeholder rate; age-specific values are set during age stratification, because
    # relapse competes with age-specific background mortality.
    model.add_transition_flow("tx_relapse", 1.0, "treatment", "subclin_lowinf")


def _treatment_success_func(tv_params: dict, lever):
    """Time-variant treatment success proportion, with any programmatic lever applied."""
    if "tx_success_pct" not in tv_params:
        raise KeyError("tv_params must include 'tx_success_pct'")
    series = tv_params["tx_success_pct"]
    tsr = stf.get_linear_interpolation_function(
        series.index.to_list(), (series / 100.0).to_list()
    )
    if lever is not None and lever.tx_success_rel != 1.0:
        tsr = Function(_capped_proportion, [tsr * lever.multiplier_func(lever.tx_success_rel)])
    return tsr


def _capped_proportion(value):
    """Keep a proportion within [0, 1] after a multiplier has been applied."""
    return jnp.clip(value, 0.0, 1.0)


def get_neg_tx_outcome_funcs(death_rate_funcs: dict, tsr) -> dict:
    """Age-specific relapse and death rates for unsuccessful treatment.

    Treatment outcomes must sum to one. Deaths from background causes during treatment are
    computed first and subtracted from the requested death proportion, so that the overall
    death proportion matches the target rather than exceeding it; the remainder becomes
    relapse.
    """

    def outcomes_for_age(background_death_rate, tsr_value, tx_duration, pct_neg_tx_death):
        prop_background_death = 1.0 - jnp.exp(-tx_duration * background_death_rate)
        requested_death_prop = (1.0 - tsr_value) * pct_neg_tx_death / 100.0
        tb_death_prop = jnp.max(
            jnp.array((requested_death_prop - prop_background_death, 0.0))
        )
        relapse_prop = 1.0 - tsr_value - tb_death_prop - prop_background_death
        return {
            "relapse": relapse_prop / tx_duration,
            "death": tb_death_prop / tx_duration,
        }

    return {
        age: Function(
            outcomes_for_age,
            [func, tsr, Parameter("tx_duration"), Parameter("pct_neg_tx_death")],
        )
        for age, func in death_rate_funcs.items()
    }


# --------------------------------------------------------------------------------------
# Births and deaths
# --------------------------------------------------------------------------------------


def _add_births_and_deaths(
    model,
    config: ModelConfig,
    total_pop: pd.Series,
    death_rate_funcs: dict,
    neg_tx_outcome_funcs: dict,
    flows: FlowRegistry,
) -> None:
    """Mortality as recycling into the youngest stratum, plus net population growth.

    All deaths are modelled as transitions back to ``mtb_naive`` in the youngest age group.
    On its own this holds population size constant; net growth is then added as an
    importation flow matching the observed trajectory. Deaths stay within their own
    non-age stratum, so stratum composition is preserved.
    """
    age_groups = list(config.age_groups)
    youngest = age_groups[0]

    for compartment in COMPARTMENTS:
        for i_age, age in enumerate(age_groups):
            # Deaths from the youngest naive compartment would be a self-loop.
            if i_age == 0 and compartment == MTB_NAIVE:
                continue
            name = f"all_cause_mortality_from_{compartment}_age_{age}"
            model.add_transition_flow(
                name=name,
                fractional_rate=death_rate_funcs[age],
                source=compartment,
                source_strata={"age": age},
                dest=MTB_NAIVE,
                dest_strata={"age": youngest},
            )
            flows.natural_death.append(name)

    # TB mortality applies to symptomatic disease only.
    for infectiousness in ["inf", "lowinf"]:
        for age in age_groups:
            name = f"tb_mortality_{infectiousness}_age_{age}"
            model.add_transition_flow(
                name=name,
                fractional_rate=Parameter(f"tb_mortality_rate_{infectiousness}"),
                source=f"clin_{infectiousness}",
                source_strata={"age": age},
                dest=MTB_NAIVE,
                dest_strata={"age": youngest},
            )
            flows.tb_death.append(name)

    for age in age_groups:
        name = f"tx_death_age_{age}"
        model.add_transition_flow(
            name=name,
            fractional_rate=neg_tx_outcome_funcs[age]["death"],
            source="treatment",
            source_strata={"age": age},
            dest=MTB_NAIVE,
            dest_strata={"age": youngest},
        )
        flows.tb_death.append(name)

    _add_births(model, config, total_pop)


def _add_births(model, config: ModelConfig, total_pop: pd.Series) -> None:
    """Net population growth, split across every active non-age stratification.

    Births are distributed according to each stratification's population split, so that the
    composition of the population stays constant over time. Splitting explicitly, one
    importation flow per stratum combination, is what makes this work for an arbitrary set
    of stratifications rather than only for the hard-coded reachability split of the
    published model.
    """
    annual_index = pd.Index(range(int(total_pop.index.min()), int(total_pop.index.max()) + 1))
    interpolated = total_pop.reindex(annual_index).interpolate()
    net_entry = interpolated.diff().dropna()

    # No growth before the first data point, then follow the observed trajectory.
    entry_rate = stf.get_sigmoidal_interpolation_function(
        [net_entry.index.min() - 1] + net_entry.index.to_list(),
        [0.0] + net_entry.to_list(),
    )

    youngest = list(config.age_groups)[0]
    strats = list(config.stratifications)

    if not strats:
        model.add_importation_flow(
            "births", entry_rate, dest=MTB_NAIVE, split_imports=False,
            dest_strata={"age": youngest},
        )
        return

    # One flow per combination of strata, weighted by the product of the splits.
    combinations = product(*[[(s.name, stratum) for stratum in s.strata] for s in strats])
    splits = {s.name: s.population_split for s in strats}
    for combination in combinations:
        dest_strata = {"age": youngest}
        proportion = 1.0
        label_parts = []
        for strat_name, stratum in combination:
            dest_strata[strat_name] = stratum
            proportion = proportion * splits[strat_name][stratum]
            label_parts.append(f"{strat_name}_{stratum}")
        model.add_importation_flow(
            f"births_{'_'.join(label_parts)}",
            entry_rate * proportion,
            dest=MTB_NAIVE,
            split_imports=False,
            dest_strata=dest_strata,
        )


def tanh_based_scaleup(
    t: float,
    shape: float,
    inflection_time: float,
    start_asymptote: float,
    end_asymptote: float = 1.0,
) -> float:
    """Smooth transition between two asymptotes, based on the hyperbolic tangent.

    ``f(t) = (tanh(shape * (t - inflection_time)) / 2 + 0.5) * (end - start) + start``

    Used for the historical scale-up of passive case detection, where the shape parameter
    controls steepness and the inflection time the midpoint of the transition.
    """
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * (
        end_asymptote - start_asymptote
    ) + start_asymptote
