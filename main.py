from irsystem import *
from utils.arguments import get_common_args

if __name__ == '__main__':
    args = get_common_args()
    ir_sys = IRSystem(args)
    ir_sys.run()
    
    #eval()
    #opt = select_model_by_ngct_10()
    
    #ir_sys.test(opt)
    #eval(args.test_dir, [args.sys_name])