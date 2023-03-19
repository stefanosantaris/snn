import argparse

class ArgumentParser():
    @staticmethod
    def parse():

        # Setup arguments parser
        parser = argparse.ArgumentParser()

        parser.add_argument("--debug", type=bool, default=False)
        parser.add_argument("--batch_size", type=int, default=2048)
        parser.add_argument("--dataset", type=str, default="criteo")
        parser.add_argument("--num_inputs", type=int, default=29)
        parser.add_argument("--num_hidden", type=int, default=6079)
        parser.add_argument("--num_outputs", type=int, default=1)
        parser.add_argument("--num_steps", type=int, default=1)
        parser.add_argument("--beta", type=float, default=0.5)
        parser.add_argument("--num_epochs", type=int, default=100)
        parser.add_argument("--plot", type=bool, default=False)
        return parser.parse_args()
