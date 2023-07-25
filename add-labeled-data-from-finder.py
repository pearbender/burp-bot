import os
import shutil

labeled_burp_dir = './burp-found-temp'
target_burps_dir = './burps-audio'
target_not_burps_dir = './not-burps-audio'

burps_label = 'burps'
not_burps_label = 'non-burps'

def main():
    if not os.path.exists(target_burps_dir):
        os.makedirs(target_burps_dir)

    if not os.path.exists(target_not_burps_dir):
        os.makedirs(target_not_burps_dir)

    new_burps = os.path.join(labeled_burp_dir, burps_label)
    new_not_burps = os.path.join(labeled_burp_dir, not_burps_label)

    if not os.path.exists(labeled_burp_dir):
        print(f"Source {labeled_burp_dir} doesn't exist.")
    
    if os.path.exists(new_burps):
        burps_files = [os.path.join(new_burps, file) for file in os.listdir(
            new_burps) if file.lower().endswith(".wav")]
        
        print(f"\nAdding {len(burps_files)} new burps...")
        exist_counter = 0
        moved_counter = 0
        for file in burps_files:
            new_file = os.path.join(target_burps_dir, os.path.basename(file))
            if os.path.exists(new_file):
                os.remove(file)
                exist_counter += 1
            else:
                shutil.move(file, new_file)
                print(new_file)
                moved_counter += 1

        print(f"\n== Added {moved_counter} new burps, skipped {exist_counter} ==")

    if os.path.exists(new_not_burps):
        not_burps_files = [os.path.join(new_not_burps, file) for file in os.listdir(
            new_not_burps) if file.lower().endswith(".wav")]
        
        print(f"\nAdding {len(not_burps_files)} new not-burps...")
        exist_counter = 0
        moved_counter = 0
        for file in not_burps_files:
            new_file = os.path.join(target_not_burps_dir, os.path.basename(file))
            if os.path.exists(new_file):
                os.remove(file)
                exist_counter += 1
            else:
                shutil.move(file, new_file)
                print(new_file)
                moved_counter += 1

        print(f"\n== Added {moved_counter} new not-burps, skipped {exist_counter} ==")


if __name__ == "__main__":
    main()