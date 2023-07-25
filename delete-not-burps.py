import os

if __name__ == "__main__":
    blacklist = {}
    with open("blacklist.txt", "r") as f:
        for line in f:
            parts = line.split("_")
            vod = parts[0]
            seconds = int(parts[1])
            if not vod in blacklist:
                blacklist[vod] = []
            blacklist[vod].append(seconds)

    for filename in os.listdir("burps"):
        if not filename.endswith(".mp4"):
            continue

        parts = filename.split("_")
        vod = parts[0]
        seconds = int(parts[1])

        if not vod in blacklist:
            continue

        for s in blacklist[vod]:
            if abs(s - seconds) <= 4:
                file_path = os.path.join("burps", filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
