#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, csv
import numpy as np
from scipy.interpolate import interp1d

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 merge_force_altitude_v3_prep.py <SmartJoint_Data.csv> <Original_TouchData.csv>")
        return

    robot_csv_path = sys.argv[1]
    original_csv_path = sys.argv[2]
    output_csv_path = "Final_Robot_Data_Synced_Prep.csv"

    print("--- データ統合 V3 (Air時先読み機能付き) ---")

    # 1. ロボットデータの読み込み
    print(f"Loading Robot Data: {robot_csv_path}")
    robot_rows = []
    with open(robot_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        robot_fieldnames = reader.fieldnames
        for row in reader:
            robot_rows.append(row)

    # 2. 元データの読み込みとストローク分割
    print(f"Loading Original Data: {original_csv_path}")
    human_strokes = {} 
    current_stroke_idx = 1
    temp_data = {'time': [], 'alt': [], 'force': []}
    
    with open(original_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ph = row['Phase'].strip()
                alt = float(row['AltitudeAngle'])
                frc = float(row['Force'])
                ts = float(row['Timestamp'])

                if ph in ['START', 'MOVE']:
                    temp_data['time'].append(ts)
                    temp_data['alt'].append(alt)
                    temp_data['force'].append(frc)
                elif ph == 'END':
                    temp_data['time'].append(ts)
                    temp_data['alt'].append(alt)
                    temp_data['force'].append(frc)
                    if temp_data['time']:
                        human_strokes[current_stroke_idx] = temp_data
                        current_stroke_idx += 1
                    temp_data = {'time': [], 'alt': [], 'force': []}
            except: pass
            
    if temp_data['time']:
        human_strokes[current_stroke_idx] = temp_data

    # 3. ロボットデータのストローク区間特定
    robot_stroke_intervals = {} 
    for idx, row in enumerate(robot_rows):
        status = row['OriginalStatus']
        if "DRAWING_STROKE_" in status:
            try:
                s_num = int(status.split('_')[-1])
                if s_num not in robot_stroke_intervals:
                    robot_stroke_intervals[s_num] = []
                robot_stroke_intervals[s_num].append(idx)
            except: pass

    # 初期化 (いったん全て0で埋める)
    for row in robot_rows:
        row['AltitudeAngle'] = "0.00000"
        row['Force'] = "0.00000"

    # --- 4. 描画部分（Pen）のマッピング ---
    print("Mapping Drawing Strokes...")
    for s_num, indices in robot_stroke_intervals.items():
        if s_num not in human_strokes: continue

        h_data = human_strokes[s_num]
        h_times = np.array(h_data['time'])
        h_alts = np.array(h_data['alt'])
        h_forces = np.array(h_data['force'])

        if len(h_times) < 2: continue

        # 正規化と補間
        h_duration = h_times[-1] - h_times[0]
        if h_duration <= 0: h_duration = 0.0001
        h_norm_t = (h_times - h_times[0]) / h_duration
        func_alt = interp1d(h_norm_t, h_alts, kind='linear', fill_value="extrapolate")
        func_force = interp1d(h_norm_t, h_forces, kind='linear', fill_value="extrapolate")

        r_start_time = float(robot_rows[indices[0]]['Timestamp'])
        r_end_time = float(robot_rows[indices[-1]]['Timestamp'])
        r_duration = r_end_time - r_start_time
        if r_duration <= 0: r_duration = 0.0001

        for r_idx in indices:
            r_current_time = float(robot_rows[r_idx]['Timestamp'])
            r_norm_pos = (r_current_time - r_start_time) / r_duration
            r_norm_pos = max(0.0, min(1.0, r_norm_pos))

            mapped_alt = float(func_alt(r_norm_pos))
            mapped_force = float(func_force(r_norm_pos))
            robot_rows[r_idx]['AltitudeAngle'] = f"{mapped_alt:.5f}"
            robot_rows[r_idx]['Force'] = f"{mapped_force:.5f}"

    # --- 5. 移動部分（Air）の先読みマッピング ---
    print("Filling Air segments with Next Stroke's Start Value...")
    
    # 次に目指すべきストロークID（最初は1）
    next_target_stroke_id = 1
    
    for row in robot_rows:
        status = row['OriginalStatus']
        
        # もし現在が描画中なら
        if "DRAWING_STROKE_" in status:
            try:
                current_id = int(status.split('_')[-1])
                # このストロークが終わったら、次は +1 を目指す
                next_target_stroke_id = current_id + 1
            except: pass
            
            # 描画中の行は、すでに Step 4 で正確な値が入っているのでスキップ
            continue
            
        else:
            # ここは Air (移動中)
            # 次のターゲットが存在するか確認
            if next_target_stroke_id in human_strokes:
                # ターゲットストロークの「開始時の値」を取得
                start_alt = human_strokes[next_target_stroke_id]['alt'][0]
                start_force = human_strokes[next_target_stroke_id]['force'][0]
                
                # Airの行に「次の準備値」を埋め込む
                row['AltitudeAngle'] = f"{start_alt:.5f}"
                row['Force'] = f"{start_force:.5f}"
            else:
                # 次がない（全工程終了後の移動）は 0.0 にする
                row['AltitudeAngle'] = "0.00000"
                row['Force'] = "0.00000"

    # 6. 保存
    output_headers = robot_fieldnames + ['AltitudeAngle', 'Force']
    
    with open(output_csv_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=output_headers)
        writer.writeheader()
        writer.writerows(robot_rows)

    print(f"統合完了: {output_csv_path}")
    print("※ 移動中(Air)は、次のストロークの開始値を維持しています。")

if __name__ == '__main__':
    main()
