import wandb
import time


class Logger:
    def __init__(self):
        self.is_init = False

    def log(self, args, dict):

        
        if args.use_wandb :
            if self.is_init is False:
                wandb.init(project=args.project_name,
                            name=f"{args.local_time}_{args.description}")
                wandb.config.update(args)
                self.is_init = True

            wandb.log(dict)


web_logger = Logger()
