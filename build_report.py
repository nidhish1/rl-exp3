"""Generate measured-result tables, task pages, and exact code excerpts."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'report_parts'
OUT.mkdir(exist_ok=True)
summary = json.loads((ROOT/'results/summary.json').read_text())
records = {t: [json.loads((ROOT/f'results/task{t}_seed{s:02d}.json').read_text())
               for s in range(10)] for t in 'ABC'}
audit = json.loads((ROOT/'results/obstacle_audit.json').read_text())
contacts = {r['seed']: r['box_obstacle_contact_steps'] for r in audit}
free = sum(r['success'] and contacts[r['seed']] == 0 for r in records['C'])

source = (ROOT/'mppi_push.py').read_text()
body = source[source.index("    p = planner.params\n", source.index('def mppi_step')):]
parts = {'sampling.py': body[:body.index('\n    # Sensor-based')],
         'update.py': body[body.index('    weights = np.exp'):body.index('\n    info =')]}
for name, code in parts.items():
    (OUT/name).write_text('\n'.join(s[4:] if s.startswith('    ') else s
                                   for s in code.splitlines())+'\n')

table = [r'\begin{center}\begin{tabular}{lrrrr}\toprule',
         r'Task & Success & Rate & Mean position error & Mean yaw error\\\midrule']
for t in 'ABC':
    s = summary[t]
    yaw = '--' if t != 'B' else f"{s['mean_yaw_error_deg']:.2f}" + r'$^\circ$'
    table.append(f"{t} & {s['successes']}/10 & {100*s['success_rate']:.0f}" +
                 r'\%' + f" & {s['mean_position_error_cm']:.2f} cm & {yaw}" + r'\\')
table += [r'\bottomrule\end{tabular}\end{center}']
(OUT/'summary.tex').write_text('\n'.join(table))
facts = (f"The hold-verified counts are " +
         ', '.join(f"{t}: {summary[t]['held_successes']}/10" for t in 'ABC') + '. ' +
         f"Task C has {free}/10 successes with no recorded box--cylinder contact; " +
         f"{sum(v>0 for v in contacts.values())}/10 episodes have such contact. " +
         "This audit is separate from the template's goal-reaching test.\n\n" +
         "Mean planning times are " +
         ', '.join(f"{t}: {summary[t]['mean_planning_ms']:.1f} ms" for t in 'ABC') + '.')
(OUT/'facts.tex').write_text(facts)

titles = {'A': 'push to a position', 'B': 'push to a position and orientation',
          'C': 'push around the obstacle'}
for task in 'ABC':
    rr, ss = records[task], summary[task]
    chosen = next((r for r in rr if r['success']), rr[0])
    seed = chosen['seed']
    z = np.load(ROOT/f'results/task{task}_seed{seed:02d}.npz')
    times = [z['t'][int(round(f*(len(z['t'])-1)))] for f in [0,.5,1]]
    text = '\\newpage\n\\section*{Question 3 -- Task '+task+': '+titles[task]+'}\n'
    text += (r'\textbf{Question.} Show plots and videos illustrating the solution, '
             'and explain the examples chosen.\n\n')
    text += f"\\textbf{{Answer.}} Task {task} achieved {ss['successes']}/10 successes "
    text += f"({100*ss['success_rate']:.0f}\\%). Screenshots and the path plot use seed {seed}, "
    text += ('the first successful seed in numeric order. ' if chosen['success']
             else 'seed 0 because no run succeeded. ')
    text += ('This is a reproducible illustrative choice; the error plots and table '
             'include every run.\n\n\\begin{center}\n')
    for label in ['start','middle','final']:
        text += f"\\includegraphics[width=.325\\textwidth]{{screenshots/task{task}_seed{seed:02d}_{label}.png}}\n"
    text += '\\\\[-2pt]{\\small ' + r' \quad '.join(
        f'{label}: {t:.2f} s' for label,t in zip(['Start','Middle','Final'],times))
    text += '}\n\\end{center}\n'
    text += f"\\includegraphics[width=\\textwidth]{{figures/task{task}_results.pdf}}\n"
    text += "\n\\textbf{Output: all ten episodes.} Errors are measured at the episode's end.\n"
    text += '\\begin{center}\\small\n\\begin{tabular}{crrrrr}\\toprule\n'
    last_col = 'Box/cylinder steps' if task == 'C' else r'Max tilt ($^\circ$)'
    text += r'Seed & Success & Position (cm) & Yaw ($^\circ$) & Time (s) & '
    text += last_col + '\\\\\\midrule\n'
    for row in rr:
        yaw = '--' if row['e_yaw'] is None else f"{np.degrees(row['e_yaw']):.2f}"
        last = str(contacts[row['seed']]) if task == 'C' else f"{row['max_tilt_deg']:.1f}"
        text += (f"{row['seed']} & {'Yes' if row['success'] else 'No'} & "
                 f"{100*row['e_pos']:.2f} & {yaw} & {row['T_sim']:.2f} & {last}" + '\\\\\n')
    text += '\\bottomrule\\end{tabular}\\end{center}\n'
    if task == 'A':
        text += ('The box moves from $(0.55,0.10)$ toward $(0.55,0.30)$ m. '
                 'The position curves show convergence; the tilt curves check stability. '
                 'Box yaw is not scored. Maximum tilt may exceed the success threshold '
                 'during motion; the template requires uprightness when scoring success.\n')
    elif task == 'B':
        failures = [str(r['seed']) for r in rr if not r['success']]
        text += ('The goal position and signed yaw vary with the seed. Separate position '
                 'and yaw plots are necessary: reaching the location alone is not success. ')
        if failures:
            text += ('Seeds '+', '.join(failures)+' fail the yaw tolerance despite reaching '
                     'the position tolerance. Their residual yaw errors indicate difficulty '
                     'making a final corrective rotation with contact-based pushing.\n')
    else:
        text += (f'The guide points route the box to the left of the cylinder. '
                 f'{free}/10 runs reach the goal without recorded box--cylinder contact. '
                 'Contact counts are reconstructed with MuJoCo at every logged pose. '
                 'The penalty is soft, so this is an audit, not a collision-avoidance guarantee. '
                 'Unsuccessful runs stall at a route transition or outside the final position tolerance.\n')
        text += f"Arm--table contacts: {ss['arm_table_steps']} steps; boxes off the table: {ss['off_table']}.\n"
    text += f"\n\\textbf{{Video:}} \\href{{videos/task{task}_seed{seed:02d}.mp4}}{{\\texttt{{task{task}\\_seed{seed:02d}.mp4}}}} "
    text += '(simulation time, plus a one-second final hold).\n'
    failures = [r for r in rr if not r['success']]
    if failures and failures[0]['seed'] != seed:
        fs = failures[0]['seed']
        text += f"\\par Failure comparison: \\href{{videos/task{task}_seed{fs:02d}.mp4}}{{\\texttt{{task{task}\\_seed{fs:02d}.mp4}}}}.\n"
    (OUT/f'task{task}.tex').write_text(text)
print('Generated report_parts from all 30 measured episodes.')
