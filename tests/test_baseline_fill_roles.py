"""A satellite filling a geometric vacancy contributes to baseline coverage."""

import pandas as pd
import pytest

import static_app as app


def plane_rows(filler_angle=312.564, filler_slot='A17'):
    rows = [dict(SVN=f'BASE{i}', PRN=f'E{i:02}', Slot=f'A{i:02}', Type='Regular',
                 Status='USABLE', Plot_Angle=(i - 1) * 45.0)
            for i in range(1, 8)]
    rows.append(dict(SVN='GSAT0232', PRN='E16', Slot=filler_slot,
                     Type='Spare' if filler_slot == 'Unk' else 'Auxiliary',
                     Status='USABLE', Plot_Angle=filler_angle))
    return pd.DataFrame(rows)


def classify(frame, **kwargs):
    return app.apply_baseline_fill_labels(frame, app.GAL_NOMINAL_SLOT_ANGLES, 8,
                                          'Galileo', plane_num=1, **kwargs)


def test_filler_counts_as_baseline_and_preserves_roster_identity():
    original = plane_rows()
    result = classify(original)
    filler = result.set_index('SVN').loc['GSAT0232']
    assert filler['Slot'] == 'A17'
    assert filler['Roster_Type'] == 'Auxiliary'
    assert filler['Type'] == 'Regular'
    assert filler['Baseline_Fill_Label'] == 'A08'
    assert filler['Marker_Label'] == 'A08'
    assert filler['Baseline_Fill_Deviation'] == pytest.approx(-2.436)
    regular = (result['Type'] == 'Regular').sum()
    reserve = result['Type'].isin(['Auxiliary', 'Spare']).sum()
    assert app.format_satellite_mix(regular, reserve)[0] == '8'
    assert original.iloc[-1]['Type'] == 'Auxiliary'
    assert 'Roster_Type' not in original
    pd.testing.assert_frame_equal(classify(result), result)


def test_extra_is_not_promoted_when_baseline_position_is_already_occupied():
    frame = plane_rows()
    frame.loc[len(frame)] = dict(SVN='BASE8', PRN='E24', Slot='A08', Type='Regular',
                                Status='USABLE', Plot_Angle=315.0)
    result = classify(frame)
    assert result.iloc[7]['Type'] == 'Auxiliary'
    assert result['Baseline_Fill_Label'].isna().all()
    assert result['Type'].value_counts().to_dict() == {'Regular': 8, 'Auxiliary': 1}


@pytest.mark.parametrize('angle,status', [(340.0, 'USABLE'), (315.0, 'NOT_USABLE')])
def test_far_or_unusable_satellite_does_not_fill_a_vacancy(angle, status):
    frame = plane_rows(filler_angle=angle)
    frame.loc[7, 'Status'] = status
    result = classify(frame)
    assert result.loc[7, 'Type'] == 'Auxiliary'
    assert pd.isna(result.loc[7, 'Baseline_Fill_Label'])


def test_unknown_slot_spare_fills_only_its_geometric_plane():
    result = classify(plane_rows(filler_slot='Unk'))
    assert result.loc[7, 'Type'] == 'Regular'
    assert result.loc[7, 'Baseline_Fill_Label'] == 'A08'


def test_competing_extras_fill_each_vacancy_only_once():
    frame = plane_rows(filler_angle=314)
    frame.loc[len(frame)] = dict(SVN='SECOND', PRN='E99', Slot='A12', Type='Auxiliary',
                                Status='USABLE', Plot_Angle=310)
    result = classify(frame)
    assert result['Baseline_Fill_Label'].notna().sum() == 1
    assert result['Type'].value_counts().to_dict() == {'Regular': 8, 'Auxiliary': 1}


def test_gps_keeps_its_separate_role_classification():
    original = plane_rows()
    result = app.apply_baseline_fill_labels(original, app.GAL_NOMINAL_SLOT_ANGLES, 8,
                                            'GPS', plane_num=1)
    assert result['Type'].equals(original['Type'])
    assert result['Baseline_Fill_Label'].isna().all()
