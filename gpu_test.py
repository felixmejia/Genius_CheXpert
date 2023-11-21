def chexpert(args, architecture, num_gpu):

        
        # overwrite args from config
        if args.load_config: args.__dict__.update(load_json(args.load_config))

        # set up output folder
        if not args.output_dir:
            if args.restore: raise RuntimeError('Must specify `output_dir` argument')
            args.output_dir: args.output_dir = os.path.join('results', time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
        # make new folders if they don't exist
        args.output_dir = args.output_dir  + str(num_gpu-1)
        writer = SummaryWriter(logdir=args.output_dir)  # creates output_dir
        if not os.path.exists(os.path.join(args.output_dir, 'vis')): os.makedirs(os.path.join(args.output_dir, 'vis'))
        if not os.path.exists(os.path.join(args.output_dir, 'plots')): os.makedirs(os.path.join(args.output_dir, 'plots'))
        if not os.path.exists(os.path.join(args.output_dir, 'best_checkpoints')): os.makedirs(os.path.join(args.output_dir, 'best_checkpoints'))

        # save config
        if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
        writer.add_text('config', str(args.__dict__))

        args.device = torch.device('cuda:{}'.format(num_gpu-1)) #if args.cuda is not None and torch.cuda.is_available() else 'cpu')
        if args.seed:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        # load model
        n_classes = len(ChexpertSmall.attr_names)
        if args.model=='NetWork':
            arch = utils.decode_arch(architecture)
            model = Network(arch, n_classes)
            model = model.to(args.device)


            # 2. init output layer with default torchvision init
            nn.init.constant_(model.classifier.bias, 0)
            # 3. store locations of forward and backward hooks for grad-cam
            grad_cam_hooks = {'forward': model.features, 'backward': model.classifier}
            # 4. init optimizer and scheduler
            optimizer = torch.optim.SGD(
                model.parameters(),
                0.1,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True
                )
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #        scheduler = None
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.n_epochs))
    #        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    #        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40000, 60000])
        elif args.model=='densenet121':
            model = densenet121(pretrained=args.pretrained).to(args.device)
            # 1. replace output layer with chexpert number of classes (pretrained loads ImageNet n_classes)
            model.classifier = nn.Linear(model.classifier.in_features, out_features=n_classes).to(args.device)
            # 2. init output layer with default torchvision init
            nn.init.constant_(model.classifier.bias, 0)
            # 3. store locations of forward and backward hooks for grad-cam
            grad_cam_hooks = {'forward': model.features.norm5, 'backward': model.classifier}
            # 4. init optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = None
    #        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    #        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40000, 60000])
        elif args.model=='aadensenet121':
            model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=n_classes,
                            attn_params={'k': 0.2, 'v': 0.1, 'nh': 8, 'relative': True, 'input_dims': (320,320)}).to(args.device)
            grad_cam_hooks = {'forward': model.features, 'backward': model.classifier}
            attn_hooks = [model.features.transition1.conv, model.features.transition2.conv, model.features.transition3.conv]
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40000, 60000])
        elif args.model=='resnet152':
            model = resnet152(pretrained=args.pretrained).to(args.device)
            model.fc = nn.Linear(model.fc.in_features, out_features=n_classes).to(args.device)
            grad_cam_hooks = {'forward': model.layer4, 'backward': model.fc}
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = None
        elif args.model=='aaresnet152':  # resnet50 layers [3,4,6,3]; resnet101 layers [3,4,23,3]; resnet 152 layers [3,8,36,3]
            model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=n_classes,
                            attn_params={'k': 0.2, 'v': 0.1, 'nh': 8, 'relative': True, 'input_dims': (320,320)}).to(args.device)
            grad_cam_hooks = {'forward': model.layer4, 'backward': model.fc}
            attn_hooks = [model.layer2[i].conv2 for i in range(len(model.layer2))] + \
                        [model.layer3[i].conv2 for i in range(len(model.layer3))] + \
                        [model.layer4[i].conv2 for i in range(len(model.layer4))]
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = None
        elif 'efficientnet' in args.model:
            model = construct_model(args.model, n_classes=n_classes).to(args.device)
            grad_cam_hooks = {'forward': model.head[1], 'backward': model.head[-1]}
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, eps=0.001)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay_factor)
        else:
            raise RuntimeError('Model architecture not supported.')

        if args.restore and os.path.isfile(args.restore):  # restore from single file, else ensemble is handled by evaluate_ensemble
            print('Restoring model weights from {}'.format(args.restore))
            model_checkpoint = torch.load(args.restore, map_location=args.device)
            model.load_state_dict(model_checkpoint['state_dict'])
            args.step = model_checkpoint['global_step']
            del model_checkpoint
            # if training, load optimizer and scheduler too
            if args.train:
                print('Restoring optimizer.')
                optim_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'optim_' + os.path.basename(args.restore))
                optimizer.load_state_dict(torch.load(optim_checkpoint_path, map_location=args.device))
                if scheduler:
                    print('Restoring scheduler.')
                    sched_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'sched_' + os.path.basename(args.restore))
                    scheduler.load_state_dict(torch.load(sched_checkpoint_path, map_location=args.device))

        # load data
        if args.restore:
            # load pretrained flag from config -- in case forgotten e.g. in post-training evaluation
            # (images still need to be normalized if training started on an imagenet pretrained model)
            args.pretrained = load_json(os.path.join(args.output_dir, 'config.json'))['pretrained']
        train_dataloader = fetch_dataloader(args, mode='train')
        valid_dataloader = fetch_dataloader(args, mode='valid')
        vis_dataloader = fetch_dataloader(args, mode='vis')

        # setup loss function for train and eval
        loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(args.device)

        print('Loaded {} (number of parameters: {:,}; weights trained to step {})'.format(
            model._get_name(), sum(p.numel() for p in model.parameters()), args.step))
        print('Train data length: ', len(train_dataloader.dataset))
        print('Valid data length: ', len(valid_dataloader.dataset))
        print('Vis data subset: ', len(vis_dataloader.dataset))

        if args.train:
            eval_metrics = train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, args)

        if args.evaluate_single_model:
            eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
            # print('Evaluate metrics -- \n\t restore: {} \n\t step: {}:'.format(args.restore, args.step))
            # print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
            # print('Loss:\n', pprint.pformat(eval_metrics['loss']))
            save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)

        if args.evaluate_ensemble:
            assert os.path.isdir(args.restore), 'Restore argument must be directory with saved checkpoints'
            eval_metrics = evaluate_ensemble(model, valid_dataloader, loss_fn, args)
            # print('Evaluate ensemble metrics -- \n\t checkpoints path {}:'.format(args.restore))
            # print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
            # print('Loss:\n', pprint.pformat(eval_metrics['loss']))
            save_json(eval_metrics, 'eval_results_ensemble', args)

        if args.visualize:
            visualize(model, vis_dataloader, grad_cam_hooks, args)
            if attn_hooks is not None:
                for x, _, idxs in vis_dataloader:
                    model(x.to(args.device))
                    patient_ids = extract_patient_ids(vis_dataloader.dataset, idxs)
                    # visualize stored attention weights for each image
                    for i in range(len(x)): vis_attn(x, patient_ids, idxs, attn_hooks, args, i)

        if args.plot_roc:
            # load results files from output_dir
            filenames = [f for f in os.listdir(args.output_dir) if f.startswith('eval_results') and f.endswith('.json')]
            if filenames==[]: raise RuntimeError('No `eval_results` files found in `{}` to plot results from.'.format(args.output_dir))
            # load and plot each
            for f in filenames:
                plot_roc(load_json(os.path.join(args.output_dir, f)), args, 'roc_pr_' + f.split('.')[0])

        writer.close()
        A=eval_metrics['aucs']
        res = 0
        for val in A.values(): 
            res += val 
        # using len() to get total keys for mean computation 
        res = res / len(A) 
        args.device.empty_cache()
        print("Liberada memory GPU ", args.device)
        return res
   