import os
import glob
import sys

if sys.version_info < (3, 0):
    input = raw_input


def invalid():
    print("INVALID CHOICE!")


def bold_text(text):
    return "\033[1m {} \033[0m".format(text)


def test_settings_config():
    content = open("settings.cfg", 'r').read()
    paths = content.split("\n")

    for path in paths:
        os.environ[path.split("=")[0]]=path.split("=")[1].replace('"',"")

    if not "MILDNET_JOB_DIR" in os.environ or not os.environ["MILDNET_JOB_DIR"]:
        print("Job directory is needed to set in settings.cfg")
        return False
    return True


def execute():
    if not test_settings_config():
        return

    ans = input("Running locally or {} (l/{})? ".format(bold_text("remote"), bold_text("r")))
    env = "remote"
    if ans=="l" or ans=="L":
        env = "local"

    configs = {"Default Models:": glob.glob("job_configs/*.cnf")}

    for path in glob.glob("job_configs/*/*.cnf"):
        conf_type = path.split("/")[-2].replace("_"," ").title()
        if conf_type in configs:
            configs[conf_type].append(path)
        else:
            configs[conf_type] = [path]

    confs = {}
    conf_types = list(configs.keys())
    conf_types.sort()
    for conf_type in conf_types:
        print(bold_text(conf_type))
        for path in configs[conf_type]:
            confs[str(len(confs)+1)] = path
            print("    {}: {}".format(str(len(confs)), path.split("/")[-1]))

    ans = input("\nSelect a job config: ")
    selected_conf_path = confs.get(ans,[None,invalid])
    selected_conf = selected_conf_path.split("/")[-1].replace(".cnf", "")
    with open(selected_conf_path, "r") as f:
        print("")
        print(f.read())

    ans = input("\nConfirm the above conf for {} ({}/n): ".format(bold_text(selected_conf), bold_text("y")))
    if ans=="n" or ans=="N":
        print("\nCorrect config and re run")
    else:
        if env=="remote":
            if not os.environ["MILDNET_JOB_DIR"].startswith("gs://"):
                print("\nKindly set google cloud storage path for MILDNET_JOB_DIR config in settings.cfg")
            existing_jobs = os.popen("gsutil ls {}".format(os.environ["MILDNET_JOB_DIR"])).read()
            existing_jobs = [v.split("/")[-2] for v in existing_jobs.split("\n") if len(v)]
            new_job_name = "{}_1".format(selected_conf)
            while new_job_name in existing_jobs:
                new_job_name = "{}_{}".format(selected_conf, int(new_job_name.split("_")[-1]) + 1)
        else:
            new_job_name = selected_conf
        command = "sh gcloud.{}.run.keras.sh {} {}".format(env, selected_conf_path, new_job_name)
        print("Executing command: {}".format(command))
        os.popen(command).read()


if __name__ == '__main__':
    execute()