hyperparameter_dictionary = {
	"scikit": {
		"classification": {
			"DecisionTrees": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
				"params": {
					"ccp_alpha": {
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						},
						"default": 0.0
					},
					"class_weight": {
						"datatype": "except",
						"default": None
					},
					"criterion": {
						"datatype": "str",
						"value": [
							"gini",
							"entropy"
						],
						"default": "gini"
					},
					"max_depth": {
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						},
						"default": None
					},
					"max_features": {
						"datatype": "except",
						"default": None
					},
					"max_leaf_nodes": {
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						},
						"default": None
					},
					"min_impurity_decrease": {
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						},
						"default": 0.0
					},
					"min_samples_leaf": {
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							0.5
						],
						"default": 1
					},
					"min_samples_split": {
						"datatype": "hybrid",
						"condition": {
							"symbol": ">",
							"value": 1
						},
						"range": [
							0.0,
							0.1
						],
						"default": 2
					},
					"min_weight_fraction_leaf": {
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							0.5
						],
						"default": 0
					},
					"splitter": {
						"datatype": "str",
						"value": [
							"best",
							"random"
						],
						"default": "best"
					}
				}
			},
			"RandomForest": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
				"params": {
					"bootstrap": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"ccp_alpha": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"class_weight": {
						"default": None,
						"datatype": "except"
					},
					"criterion": {
						"default": "gini",
						"datatype": "str",
						"value": [
							"gini",
							"entropy"
						]
					},
					"max_depth": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"max_features": {
						"default": "auto",
						"datatype": "except"
					},
					"max_leaf_nodes": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"max_samples": {
						"default": None,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 1.0]
					},
					"min_impurity_decrease": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_impurity_split": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_samples_leaf": {
						"default": 1,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							0.5
						]
					},
					"min_samples_split": {
						"default": 2,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 1
						}
					},
					"min_weight_fraction_leaf": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 0.5]
					},
					"n_estimators": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"n_jobs": {
						"default": None,
						"datatype": "except"
					},
					"oob_score": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"Bagging": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html",
				"params": {
					"base_estimator": {
						"default": None,
						"datatype": "except"
					},
					"bootstrap": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"bootstrap_features": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"max_features": {
						"default": 1.0,
						"datatype": "except"
					},
					"max_samples": {
						"default": 1.0,
						"datatype": "except"
					},
					"n_estimators": {
						"default": 10,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"n_jobs": {
						"default": None,
						"datatype": "except"
					},
					"oob_score": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"ExtraTrees": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",
				"params": {
					"bootstrap": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"ccp_alpha": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"class_weight": {
						"default": None,
						"datatype": "except"
					},
					"criterion": {
						"default": "gini",
						"datatype": "str",
						"value": [
							"gini",
							"entropy"
						]
					},
					"max_depth": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"max_features": {
						"default": "auto",
						"datatype": "except"
					},
					"max_leaf_nodes": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"max_samples": {
						"default": None,
						"datatype": "except"
					},
					"min_impurity_decrease": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_impurity_split": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_samples_leaf": {
						"default": 1,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							0.5
						]
					},
					"min_samples_split": {
						"default": 2,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"min_weight_fraction_leaf": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 0.5]
					},
					"n_estimators": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"n_jobs": {
						"default": None,
						"datatype": "except"
					},
					"oob_score": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"KNearestNeighbour": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
				"params": {
					"algorithm": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"ball_tree",
							"kd_tree",
							"brute"
						]
					},
					"leaf_size": {
						"default": 30,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 1
						}
					},
					"metric": {
						"default": "minkowski",
						"datatype": "str",
						"value": [
							"euclidean",
							"manhattan",
							"chebyshev",
							"minkowski",
							"wminkowski",
							"seuclidean",
							"mahalanobis"
						]
					},
					"metric_params": {
						"default": None,
						"datatype": "except"
					},
					"n_jobs": {
						"default": None,
						"datatype": "except"
					},
					"n_neighbors": {
						"default": 5,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"p": {
						"default": 2,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"weights": {
						"default": "uniform",
						"datatype": "str",
						"value": [
							"uniform"
						]
					}
				}
			},
			"AdaBoost": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
				"params": {
					"algorithm": {
						"default": "SAMME.R",
						"datatype": "str",
						"value": [
							"SAMME",
							"SAMME.R"
						]
					},
					"base_estimator": {
						"default": None,
						"datatype": "except"
					},
					"learning_rate": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"n_estimators": {
						"default": 50,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					}
				}
			},
			"NaiveBayesMultinomial": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html",
				"params": {
					"alpha": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"class_prior": {
						"default": None,
						"datatype": "except"
					},
					"fit_prior": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					}
				}
			},
			"GradientBoostingMachines": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
				"params": {
					"ccp_alpha": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"criterion": {
						"default": "friedman_mse",
						"datatype": "str",
						"value": [
							"friedman_mse",
							"mse",
							"mae"
						]
					},
					"init": {
						"default": None,
						"datatype": "except"
					},
					"learning_rate": {
						"default": 0.1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"loss": {
						"default": "deviance",
						"datatype": "str",
						"value": [
							"deviance",
							"exponential"
						]
					},
					"max_depth": {
						"default": 3,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"max_features": {
						"default": None,
						"datatype": "except"
					},
					"max_leaf_nodes": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"min_impurity_decrease": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_impurity_split": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_samples_leaf": {
						"default": 1,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							0.5
						]
					},
					"min_samples_split": {
						"default": 2,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							1.0
						]
					},
					"min_weight_fraction_leaf": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 0.5]
					},
					"n_estimators": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"n_iter_no_change": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"subsample": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 1.0]
					},
					"tol": {
						"default": 0.0001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"validation_fraction": {
						"default": 0.1,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 1.0]
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"XGradientBoosting": {"reference_link": "https://xgboost.readthedocs.io/en/latest/python/python_api.html#" ,"params":{
		"objective": {
			"default": "binary:logistic",
			"datatype": "except"
		},
		"use_label_encoder": {
			"default": True,
			"datatype": "str",
			"value": [
				"True",
				"False"
			]
		},
		"base_score": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": "<=>",
				"value": 0.0
			}
		},
		"booster": {
			"default": None,
			"datatype": "str",
			"value": [
				"gbtree",
				"gblinear",
				"dart"
			]
		},
		"colsample_bylevel": {
			"default": None,
			"datatype": "float",
			"compare_type": "range",
			"range": [
				0,
				1
			]
		},
		"colsample_bynode": {
			"default": None,
			"datatype": "float",
			"compare_type": "range",
			"range": [
				0,
				1
			]
		},
		"colsample_bytree": {
			"default": None,
			"datatype": "float",
			"compare_type": "range",
			"range": [
				0,
				1
			]
		},
		"gamma": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"gpu_id": {
			"default": None,
			"datatype": "except"
		},
		"importance_type": {
			"default": "gain",
			"datatype": "str",
			"value": [
				"gain",
				"weight",
				"cover",
				"total_gain",
				"total_co"
			]
		},
		"interaction_constraints": {
			"default": None,
			"datatype": "except"
		},
		"learning_rate": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"max_delta_step": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"max_depth": {
			"default": None,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">",
				"value": 0
			}
		},
		"min_child_weight": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"missing": {
			"default": None,
			"datatype": "except"
		},
		"monotone_constraints": {
			"default": None,
			"datatype": "except"
		},
		"n_estimators": {
			"default": 100,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">",
				"value": 0
			}
		},
		"n_jobs": {
			"default": None,
			"datatype": "except"
		},
		"num_parallel_tree": {
			"default": None,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 1
			}
		},
		"random_state": {
			"default": None,
			"datatype": "except"
		},
		"reg_alpha": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"reg_lambda": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"scale_pos_weight": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"subsample": {
			"default": None,
			"datatype": "float",
			"compare_type": "range",
			"range": [
				0,
				1
			]
		},
		"tree_method": {
			"default": None,
			"datatype": "except"
		},
		"validate_parameters": {
			"default": None,
			"datatype": "str",
			"value": [
				"True",
				"False"
			]
		},
		"verbosity": {
			"default": None,
			"datatype": "except"
		}
	}},
			"SupportVectorMachines": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
				"params": {
					"C": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"break_ties": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"cache_size": {
						"default": 200,
						"datatype": "except"
					},
					"class_weight": {
						"default": None,
						"datatype": "except"
					},
					"coef0": {
						"default": 0.0,
						"datatype": "except"
					},
					"decision_function_shape": {
						"default": "ovr",
						"datatype": "str",
						"value": [
							"ovo",
							"ovr"
						]
					},
					"degree": {
						"default": 3,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"gamma": {
						"default": "scale",
						"datatype": "str",
						"value": [
							"scale",
							"auto"
						]
					},
					"kernel": {
						"default": "rbf",
						"datatype": "str",
						"value": [
							"linear",
							"poly",
							"rbf",
							"sigmoid",
							"precomputed"
						]
					},
					"max_iter": {
						"default": -1,
						"datatype": "int",
						"compare_type": "<=>"
					},
					"probability": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"shrinking": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"tol": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"verbose": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					}
				}
			},
			"LogisticRegression": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
				"params": {
					"C": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"class_weight": {
						"default": None,
						"datatype": "except"
					},
					"dual": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"fit_intercept": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"intercept_scaling": {
						"default": 1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"l1_ratio": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"max_iter": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"multi_class": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"ovr",
							"multinomial"
						]
					},
					"n_jobs": {
						"default": None,
						"datatype": "except"
					},
					"penalty": {
						"default": "l2",
						"datatype": "str",
						"value": [
							"l1",
							"l2",
							"elasticnet",
							"none"
						]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"solver": {
						"default": "lbfgs",
						"datatype": "str",
						"value": [
							"newton-cg",
							"lbfgs",
							"liblinear",
							"sag",
							"saga"
						]
					},
					"tol": {
						"default": 0.0001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			}
		},
		"regression": {
			"DecisionTrees": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
				"params": {
					"ccp_alpha": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"criterion": {
						"default": "mse",
						"datatype": "str",
						"value": [
							"mse",
							"friedman_mse",
							"mae",
							"poisson"
						]
					},
					"max_depth": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"max_features": {
						"default": None,
						"datatype": "except"
					},
					"max_leaf_nodes": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"min_impurity_decrease": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_impurity_split": {
						"default": None,
						"datatype": "except"
					},
					"min_samples_leaf": {
						"default": 1,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							0.5
						]
					},
					"min_samples_split": {
						"default": 2,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">",
							"value": 1
						},
						"range": [
							0.0,
							1.0
						]
					},
					"min_weight_fraction_leaf": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 0.5]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"splitter": {
						"default": "best",
						"datatype": "str",
						"value": [
							"best",
							"random"
						]
					}
				}
			},
			"RandomForest": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
				"params": {
					"bootstrap": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"ccp_alpha": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"criterion": {
						"default": "mse",
						"datatype": "str",
						"value": ["mse","mae"]
					},
					"max_depth": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"max_features": {
						"default": "auto",
						"datatype": "except"
					},
					"max_leaf_nodes": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"max_samples": {
						"default": None,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 1.0]
					},
					"min_impurity_decrease": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_impurity_split": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_samples_leaf": {
						"default": 1,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							0.5
						]
					},
					"min_samples_split": {
						"default": 2,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							1.0
						]
					},
					"min_weight_fraction_leaf": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 0.5]
					},
					"n_estimators": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"n_jobs": {
						"default": None,
						"datatype": "except"
					},
					"oob_score": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"Bagging": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html",
				"params": {
					"base_estimator": {
						"default": None,
						"datatype": "except"
					},
					"bootstrap": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"bootstrap_features": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"max_features": {
						"default": 1.0,
						"datatype": "except"
					},
					"max_samples": {
						"default": 1.0,
						"datatype": "except"
					},
					"n_estimators": {
						"default": 10,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"n_jobs": {
						"default": None,
						"datatype": "except"
					},
					"oob_score": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"GradientBoostingMachines": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
				"params": {
					"alpha": {
						"default": 0.9,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 1.0]
					},
					"ccp_alpha": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"criterion": {
						"default": "friedman_mse",
						"datatype": "str",
						"value": [
							"friedman_mse",
							"mse",
							"mae"
						]
					},
					"init": {
						"default": None,
						"datatype": "except"
					},
					"learning_rate": {
						"default": 0.1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"loss": {
						"default": "ls",
						"datatype": "str",
						"value": [
							"ls",
							"lad",
							"huber",
							"quantile"
						]
					},
					"max_depth": {
						"default": 3,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"max_features": {
						"default": None,
						"datatype": "except"
					},
					"max_leaf_nodes": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"min_impurity_decrease": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_impurity_split": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_samples_leaf": {
						"default": 1,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							0.5
						]
					},
					"min_samples_split": {
						"default": 2,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							1.0
						]
					},
					"min_weight_fraction_leaf": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 0.5]
					},
					"n_estimators": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"n_iter_no_change": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"subsample": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 1.0]
					},
					"tol": {
						"default": 0.0001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"validation_fraction": {
						"default": 0.1,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 1.0]
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"XGradientBoosting": {"reference_link": "https://xgboost.readthedocs.io/en/latest/python/python_api.html#" ,"params":{
		"objective": {
			"default": "binary:logistic",
			"datatype": "except"
		},
		"use_label_encoder": {
			"default": True,
			"datatype": "str",
			"value": [
				"True",
				"False"
			]
		},
		"base_score": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": "<=>",
				"value": 0.0
			}
		},
		"booster": {
			"default": None,
			"datatype": "str",
			"value": [
				"gbtree",
				"gblinear",
				"dart"
			]
		},
		"colsample_bylevel": {
			"default": None,
			"datatype": "float",
			"compare_type": "range",
			"range": [
				0,
				1
			]
		},
		"colsample_bynode": {
			"default": None,
			"datatype": "float",
			"compare_type": "range",
			"range": [
				0,
				1
			]
		},
		"colsample_bytree": {
			"default": None,
			"datatype": "float",
			"compare_type": "range",
			"range": [
				0,
				1
			]
		},
		"gamma": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"gpu_id": {
			"default": None,
			"datatype": "except"
		},
		"importance_type": {
			"default": "gain",
			"datatype": "str",
			"value": [
				"gain",
				"weight",
				"cover",
				"total_gain",
				"total_co"
			]
		},
		"interaction_constraints": {
			"default": None,
			"datatype": "except"
		},
		"learning_rate": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"max_delta_step": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"max_depth": {
			"default": None,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">",
				"value": 0
			}
		},
		"min_child_weight": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"missing": {
			"default": None,
			"datatype": "except"
		},
		"monotone_constraints": {
			"default": None,
			"datatype": "except"
		},
		"n_estimators": {
			"default": 100,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">",
				"value": 0
			}
		},
		"n_jobs": {
			"default": None,
			"datatype": "except"
		},
		"num_parallel_tree": {
			"default": None,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 1
			}
		},
		"random_state": {
			"default": None,
			"datatype": "except"
		},
		"reg_alpha": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"reg_lambda": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"scale_pos_weight": {
			"default": None,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"subsample": {
			"default": None,
			"datatype": "float",
			"compare_type": "range",
			"range": [
				0,
				1
			]
		},
		"tree_method": {
			"default": None,
			"datatype": "except"
		},
		"validate_parameters": {
			"default": None,
			"datatype": "str",
			"value": [
				"True",
				"False"
			]
		},
		"verbosity": {
			"default": None,
			"datatype": "except"
		}
	}},
			"ExtraTrees": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
				"params": {
					"bootstrap": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"ccp_alpha": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"criterion": {
						"default": "mse",
						"datatype": "str",
						"value": [
							"mse",
							"mae"
						]
					},
					"max_depth": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"max_features": {
						"default": "auto",
						"datatype": "except"
					},
					"max_leaf_nodes": {
						"default": None,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"max_samples": {
						"default": None,
						"datatype": "except"
					},
					"min_impurity_decrease": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_impurity_split": {
						"default": None,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_samples_leaf": {
						"default": 1,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							0.5
						]
					},
					"min_samples_split": {
						"default": 2,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							0.0,
							1.0
						]
					},
					"min_weight_fraction_leaf": {
						"default": 0.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0.0, 0.5]
					},
					"n_estimators": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"n_jobs": {
						"default": None,
						"datatype": "except"
					},
					"oob_score": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					},
					"verbose": {
						"default": 0,
						"datatype": "except"
					},
					"warm_start": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"AdaBoost": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html",
				"params": {
					"base_estimator": {
						"default": None,
						"datatype": "except"
					},
					"learning_rate": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"loss": {
						"default": "linear",
						"datatype": "str",
						"value": [
							"linear",
							"square",
							"exponential"
						]
					},
					"n_estimators": {
						"default": 50,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"random_state": {
						"default": None,
						"datatype": "except"
					}
				}
			},
			"SupportVectorMachines": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
				"params": {
					"C": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"cache_size": {
						"default": 200,
						"datatype": "except"
					},
					"coef0": {
						"default": 0.0,
						"datatype": "except"
					},
					"degree": {
						"default": 3,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"epsilon": {
						"default": 0.1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"gamma": {
						"default": "scale",
						"datatype": "str",
						"value": [
							"scale",
							"auto"
						]
					},
					"kernel": {
						"default": "rbf",
						"datatype": "str",
						"value": [
							"linear",
							"poly",
							"rbf",
							"sigmoid",
							"precomputed"
						]
					},
					"max_iter": {
						"default": -1,
						"datatype": "int",
						"compare_type": "<=>"
					},
					"shrinking": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"tol": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"verbose": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					}
				}
			},
			"LinearRegression": {"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html","params":{"copy_X": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},"fit_intercept":{"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]},"normalize": {"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]},"positive": {"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]}}}
		},
		"clustering": {
			"KMeansClustering": {
        		"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
				"params": {
				  "algorithm": {
					"default": "auto",
					"datatype": "str",
					"value": [
					  "auto",
					  "full",
					  "elkan"
					]
				  },
				  "copy_x": {
					"default": True,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  },
				  "init": {
					"default": "k-means++",
					"datatype": "str",
					"value": [
					  "k-means++",
					  "random"
					]
				  },
				  "max_iter": {
					"default": 300,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "n_clusters": {
					"default": 8,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "n_init": {
					"default": 10,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "tol": {
					"default": 0.0001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  },
				  "verbose": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  }
				}
			  },
			"MiniBatchKMeans": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html",
				"params": {
					"batch_size": {
			"default": 100,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">",
				"value": 0
			}
		},
		"compute_labels": {
			"default": True,
			"datatype": "str",
			"value": [
				True,
				False
			]
		},
		"init": {
			"default": "k-means++",
			"datatype": "str",
			"value": [
				"k-means++",
				"random"
			]
		},
		"init_size": {
			"default": None,
			"datatype": "except"
		},
		"max_iter": {
			"default": 100,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.5
			}
		},
		"max_no_improvement": {
			"default": 10,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"n_clusters": {
			"default": 8,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">",
				"value": 0
			}
		},
		"n_init": {
			"default": 3,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">",
				"value": 0
			}
		},
		"random_state": {
			"default": None,
			"datatype": "except"
		},
		"reassignment_ratio": {
			"default": 0.01,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"tol": {
			"default": 0.0,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": "<=>",
				"value": 0.0
			}
		},
		"verbose": {
			"default": 0.0,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": "<=>",
				"value": 0.0
			}
		}

		}
	},
			"AffinityPropagation": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html",
				"params": {
				  "affinity": {
					"default": "euclidean",
					"datatype": "str",
					"value": [
					  "euclidean"
					]
				  },
				  "convergence_iter": {
					"default": 15,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "copy": {
					"default": True,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  },
				  "damping": {
					"default": 0.5,
					"datatype": "float",
					"compare_type": "range",
					"range": [
					  0.5,
					  1
					]
				  },
				  "max_iter": {
					"default": 200,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "preference": {
					"default": None,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  },
				  "verbose": {
					"default": False,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  }
				}
			  },
			"MeanShift": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html",
				"params": {
				  "bandwidth": {
					"default": None,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0.0
					}
				  },
				  "bin_seeding": {
					"default": False,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  },
				  "cluster_all": {
					"default": True,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  },
				  "max_iter": {
					"default": 300,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "min_bin_freq": {
					"default": 1,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "n_jobs": {
					"default": None,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<>",
					  "value": 0
					}
				  }
				}
			  },
			"Birch": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html",
				"params": {
				  "branching_factor": {
					"default": 50,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 1
					}
				  },
				  "compute_labels": {
					"default": True,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  },
				  "copy": {
					"default": True,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  },
				  "n_clusters": {
					"default": 3,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "threshold": {
					"default": 0.5,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  }
				}
			  },
			"SpectralClustering": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html",
				"params": {
				  "affinity": {
					"default": "rbf",
					"datatype": "str",
					"value": [
					  "rbf",
					  "nearest_neighbors"
					]
				  },
				  "assign_labels": {
					"default": "kmeans",
					"datatype": "str",
					"value": [
					  "kmeans",
					  "discretize"
					]
				  },
				  "coef0": {
					"default": 1,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  },
				  "degree": {
					"default": 3,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  },
				  "eigen_solver": {
					"default": None,
					"datatype": "str",
					"value": [
					  "arpack",
					  "lobpcg"
					]
				  },
				  "eigen_tol": {
					"default": 0.0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  },
				  "gamma": {
					"default": 1.0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">=",
					  "value": 0.0
					}
				  },
				  "n_clusters": {
					"default": 8,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "n_components": {
					"default": None,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "n_init": {
					"default": 10,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "n_jobs": {
					"default": None,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "n_neighbors": {
					"default": 10,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "verbose": {
					"default": False,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  }
				}
			  },
			"AgglomerativeClustering": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html",
				"params": {
				  "affinity": {
					"default": "euclidean",
					"datatype": "str",
					"value": [
					  "euclidean",
					  "l1",
					  "l2",
					  "manhattan",
					  "cosine"
					]
				  },
				  "compute_distances": {
					"default": False,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  },
				  "compute_full_tree": {
					"default": False,
					"datatype": " str",
					"value": [
					  True,
					  False
					]
				  },
				  "distance_threshold": {
					"default": None,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  },
				  "linkage": {
					"default": "ward",
					"datatype": "str",
					"value": [
					  "ward",
					  "complete",
					  "average",
					  "single"
					]
				  },
				  "n_clusters": {
					"default": 2,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  }
				}
			  },
			"DBScan": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
				"params": {
				  "algorithm": {
					"default": "auto",
					"datatype": "str",
					"value": [
					  "auto",
					  "ball_tree",
					  "kd_tree",
					  "brute"
					]
				  },
				  "eps": {
					"default": 0.5,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0.0
					}
				  },
				  "leaf_size": {
					"default": 30,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "metric": {
					"default": "euclidean",
					"datatype": "str",
					"value": [
					  "euclidean"
					]
				  },
				  "min_samples": {
					"default": 5,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "n_jobs": {
					"default": None,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<>",
					  "value": 0
					}
				  },
				  "p": {
					"default": None,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  }
				}
			  },
			"Optics": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html",
				"params": {
				  "algorithm": {
					"default": "auto",
					"datatype": "str",
					"value": [
					  "auto",
					  "ball_tree",
					  "kd_tree",
					  "brute"
					]
				  },
				  "cluster_method": {
					"default": "xi",
					"datatype": "str",
					"value": [
					  "xi",
					  "dbscan"
					]
				  },
				  "eps": {
					"default": None,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  },
				  "leaf_size": {
					"default": 30,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">=",
					  "value": 1
					}
				  },
				  "max_eps": {
					"default": "inf",
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  },
				  "metric": {
					"default": "minkowski",
					"datatype": "str",
					"value": ["cityblock",
					  "cosine",
					  "euclidean",
					  "l1",
					  "l2",
					  "manhattan",
					  "braycurtis",
					  "canberra",
					  "chebyshev",
					  "correlation",
					  "dice",
					  "hamming",
					  "jaccard",
					  "kulsinski",
					  "minkowski",
					  "rogerstanimoto",
					  "russellrao",
					  "sokalmichener",
					  "sokalsneath",
					  "sqeuclidean",
					  "yule"
					]
				  },
				  "min_cluster_size": {
					"default": None,
					"datatype": "hybrid",
					"condition": {
					  "symbol": ">=",
					  "value": 1
					},
					"range": [
					  0.0,
					  1.0
					]
				  },
				  "min_samples": {
					"default": 5,
					"datatype": "hybrid",
					"condition": {
					  "symbol": ">=",
					  "value": 1
					},
					"range": [
					  0.0,
					  1.0
					]
				  },
				  "n_jobs": {
					"default": None,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<>",
					  "value": 0
					}
				  },
				  "p": {
					"default": 2,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 1
					}
				  },
				  "predecessor_correction": {
					"default": True,
					"datatype": "str",
					"value": [
					  True,
					  False
					]

				  },
				  "xi": {
					"default": 0.05,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0.0
					}
				  }
				}
			  },
			"GaussianMixtures": {
				"reference_link": "https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html",
				"params": {
				  "covariance_type": {
					"default": "full",
					"datatype": "str",
					"value": [
					  "full",
					  "tied",
					  "diag",
					  "spherical"
					]
				  },
				  "init_params": {
					"default": "kmeans",
					"datatype": "str",
					"value": [
					  "kmeans",
					  "random"
					]
				  },
				  "max_iter": {
					"default": 100,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "n_components": {
					"default": 1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "n_init": {
					"default": 1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0
					}
				  },
				  "reg_covar": {
					"default": 0.000001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 0.0
					}
				  },
				  "tol": {
					"default": 0.001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">=",
					  "value": 0.0
					}
				  },
				  "verbose": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "verbose_interval": {
					"default": 10,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<>",
					  "value": 0
					}
				  },
				  "warm_start": {
					"default": False,
					"datatype": "str",
					"value": [
					  True,
					  False
					]
				  }
				}
			  }
			}
	},
	"weka": {
		"classification":{
			"Logistic": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/functions/Logistic.html",
				"params": {
					"-S": {
						"default": False,
						"datatype": "except"
					},
					"-M": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					}
				}
			},
			"MultilayerPerceptron": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/functions/MultilayerPerceptron.html",
				"params": {
					"-L": {
						"default": 0.3,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							1.0
						]
					},
					"-M": {
						"default": 0.2,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							1.0
						]
					},
					"-N": {
						"default": 500,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-V": {
						"default": 0,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							100.0
						]
					},
					"-S": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"-E": {
						"default": 20,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-A": {
						"default": False,
						"datatype": "except"
					},
					"-B": {
						"default": False,
						"datatype": "except"
					},
					"-H": {
						"default": "a",
						"datatype": "str",
						"value": [
							"a",
							"i",
							"o",
							"t"
						]
					},
					"-C": {
						"default": False,
						"datatype": "except"
					},
					"-I": {
						"default": False,
						"datatype": "except"
					},
					"-R": {
						"default": False,
						"datatype": "except"
					},
					"-D": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"NaiveBayesMultinomial": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/bayes/NaiveBayesMultinomial.html",
				"params": {
					"-output-debug-info": {
						"default": False,
						"datatype": "except"
					},
					"-do-not-check-capabilities": {
						"default": False,
						"datatype": "except"
					},
					"-num-decimal-places": {
						"default": 2,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-batch-size": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					}
				}
			},
			"RandomForest": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/trees/RandomForest.html",
				"params": {
					"-P": {
						"default": 100,
						"datatype": "int",
						"compare_type": "range",
						"range": [0, 100]
					},
					"-O": {
						"default": False,
						"datatype": "except"
					},
					"-store-out-of-bag-predictions": {
						"default": False,
						"datatype": "except"
					},
					"-output-out-of-bag-complexity-statistics": {
						"default": False,
						"datatype": "except"
					},
					"-print": {
						"default": False,
						"datatype": "except"
					},
					"-attribute-importance": {
						"default": False,
						"datatype": "except"
					},
					"-I": {
						"default": 100,
						"datatype": "int",
						"compare_type": "range",
						"range": [64, 128]
					},
					"-num-slots": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-K": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"-M": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-V": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"-S": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-depth": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"-N": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"-U": {
						"default": False,
						"datatype": "except"
					},
					"-B": {
						"default": False,
						"datatype": "except"
					},
					"-output-debug-info": {
						"default": False,
						"datatype": "except"
					},
					"-do-not-check-capabilities": {
						"default": False,
						"datatype": "except"
					},
					"-num-decimal-places": {
						"default": 2,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-batch-size": {
						"default": 100,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					}
				}
			},
			"LibSVM": {
				"reference_link": "https://weka.sourceforge.io/doc.packages/LibSVM/weka/classifiers/functions/LibSVM.html",
				"params": {
					"-S": {
						"default": 0,
						"datatype": "int",
						"compare_type": "range",
						"range": [0, 4]
					},
					"-K": {
						"default": 2,
						"datatype": "int",
						"compare_type": "range",
						"range": [0, 3]
					},
					"-D": {
						"default": False,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-R": {
						"default": 0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"-C": {
						"default": 1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"-N": {
						"default": 0.5,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"-Z": {
						"default": False,
						"datatype": "except"
					},
					"-J": {
						"default": False,
						"datatype": "except"
					},
					"-V": {
						"default": False,
						"datatype": "except"
					},
					"-P": {
						"default": 0.1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"-M": {
						"default": 40,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-E": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"-H": {
						"default": False,
						"datatype": "except"
					},
					"-W": {
						"default": 1,
						"datatype": "except"
					},
					"-B": {
						"default": False,
						"datatype": "except"
					},
					"-seed": {
						"default": 1,
						"datatype": "except"
					}
				}
			},
			"AdaBoostM1": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/meta/AdaBoostM1.html",
				"params": {
					"-P": {
						"default": 100,
						"datatype": "int",
						"compare_type": "range",
						"range": [0, 100]
					},
					"-Q": {
						"default": False,
						"datatype": "except"
					},
					"-S": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-I": {
						"default": 10,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-D": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"AttributeSelectedClassifier": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/meta/AttributeSelectedClassifier.html",
				"params": {
					"-D": {
						"default": False,
						"datatype": "except"
					},
					"-U": {
						"default": False,
						"datatype": "except"
					},
					"-R": {
						"default": False,
						"datatype": "except"
					},
					"-B": {
						"default": False,
						"datatype": "except"
					},
					"-L": {
						"default": False,
						"datatype": "except"
					},
					"-A": {
						"default": False,
						"datatype": "except"
					}


				}
			},
			"Bagging": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/meta/Bagging.html",
				"params": {
					"-P": {
						"default": 100,
						"datatype": "int",
						"compare_type": "range",
						"range": [0, 100]
					},
					"-O": {
						"default": False,
						"datatype": "except"
					},
					"-print": {
						"default": False,
						"datatype": "except"
					},
					"-store-out-of-bag-predictions": {
						"default": False,
						"datatype": "except"
					},
					"-output-out-of-bag-complexity-statistics": {
						"default": False,
						"datatype": "except"
					},
					"-represent-copies-using-weights": {
						"default": False,
						"datatype": "except"
					},
					"-S": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-num-slots": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-I": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-D": {
						"default": False,
						"datatype": "except"
					},
					"-R": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"KStar": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/lazy/KStar.html",
				"params": {
					"-B": {
						"default": 20,
						"datatype": "int",
						"compare_type": "range",
						"range": [0, 100]
					},
					"-E": {
						"default": False,
						"datatype": "except"
					},
					"-M": {
						"default": "a",
						"datatype": "str",
						"value": [
							"a",
							"d",
							"m",
							"n"
						]
					}
				}
			},
			"DecisionTable": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/rules/DecisionTable.html",
				"params": {

					"-X": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-E": {
						"default": "acc",
						"datatype": "str",
						"value": [
							"acc",
							"rmse",
							"mae",
							"auc"
						]
					},
					"-I": {
						"default": False,
						"datatype": "except"
					},
					"-R": {
						"default": False,
						"datatype": "except"
					},
					"-P": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"IBk": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/lazy/IBk.html",
				"params": {
					"-I": {
						"default": False,
						"datatype": "except"
					},
					"-F": {
						"default": False,
						"datatype": "except"
					},
					"-K": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-E": {
						"default": False,
						"datatype": "except"
					},
					"-W": {
						"default": False,
						"datatype": "except"
					},
					"-X": {
						"default": False,
						"datatype": "except"
					}
				}
			},
 			"RandomTree": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/trees/RandomTree.html",
				"params": {
					"-K": {
						"default": 0,
						"datatype": "except"
					},
					"-M": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-V": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"-S": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-depth": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"-N": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"-U": {
						"default": False,
						"datatype": "except"
					},
					"-B": {
						"default": False,
						"datatype": "except"
					},
					"-output-debug-info": {
						"default": False,
						"datatype": "except"
					},
					"-do-not-check-capabilities": {
						"default": False,
						"datatype": "except"
					},
					"-num-decimal-places": {
						"default": 2,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					}
				}
			},
			"SMO": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/functions/SMO.html",
				"params": {
					"-no-checks": {
						"default": False,
						"datatype": "except"
					},
					"-C": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-N": {
						"default": 0,
						"datatype": "int",
						"compare_type": "range",
						"range": [0, 2]
					},
					"-L": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"-P": {
						"default": 1e-12,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"-M": {
						"default": False,
						"datatype": "except"
					},
					"-V": {
						"default": False,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 1
						}
					},
					"-W": {
						"default": 1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-output-debug-info": {
						"default": False,
						"datatype": "except"
					},
					"-do-not-check-capabilities": {
						"default": False,
						"datatype": "except"
					},
					"-num-decimal-places": {
						"default": 2,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					}
				}
			}
		},
		"regression": {
			"LinearRegression": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/functions/LinearRegression.html",
				"params": {
					"-S": {
						"default": 0,
						"datatype": "int",
						"compare_type": "range",
						"range": [0, 100]
					},
					"-C": {
						"default": False,
						"datatype": "except"
					},
					"-R": {
						"default": 8e-08,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						}
					},
					"-minimal": {
						"default": False,
						"datatype": "except"
					},
					"-additional-stats": {
						"default": False,
						"datatype": "except"
					},
					"-output-debug-info": {
						"default": False,
						"datatype": "except"
					},
					"-do-not-check-capabilities": {
						"default": False,
						"datatype": "except"
					}
				}
			},
			"AdditiveRegression": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/classifiers/meta/AdditiveRegression.html",
				"params": {
					"-S": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "range",
						"range": [0, 1]
					},
					"-I": {
						"default": 10,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"-A": {
						"default": False,
						"datatype": "except"
					},
					"-D": {
						"default": False,
						"datatype": "except"
					}
				}
			}
		},
		"clustering": {
			"MakeDensityBasedClusterer": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/clusterers/MakeDensityBasedClusterer.html",
				"params": {
				  "-M": {
					"default": 0.000001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-S": {
					"default": 10,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-V": {
					"default": False,
					"datatype": "except"
				  }
				}
			  },
			"HierarchicalClusterer": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/clusterers/HierarchicalClusterer.html",
				"params": {
				  "-L": {
					"default": "SINGLE",
					"datatype": "str",
					"value": [
					  "SINGLE",
					  "COMPLETE",
					  "AVERAGE",
					  "MEAN",
					  "CENTROID",
					  "WARD",
					  "ADJCOMPLETE",
					  "NEIGHBOR_JOINING"
					]
				  },
				  "-P": {
					"default": False,
					"datatype": "except"
				  },
				  "-D": {
					"default": False,
					"datatype": "except"
				  },
				  "-B": {
					"default": False,
					"datatype": "except"
				  }
				}
			  },
			"EM": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/clusterers/EM.html",
				"params": {
				  "-X": {
					"default": 5,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-K": {
					"default": 10,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">=",
					  "value": 1
					}
				  },
				  "-max": {
					"default": -1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-ll-cv": {
					"default": 0.000001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-I": {
					"default": 100,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">=",
					  "value": 1
					}
				  },
				  "-ll-iter": {
					"default": 0.000001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-V": {
					"default": False,
					"datatype": "except"
				  },
				  "-M": {
					"default": 0.000001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-O": {
					"default": False,
					"datatype": "except"
				  },
				  "-num-slots": {
					"default": 1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">=",
					  "value": 1
					}
				  },
				  "-S": {
					"default": 100,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-output-debug-info": {
					"default": False,
					"datatype": "except"
				  },
				  "-do-not-check-capabilities": {
					"default": False,
					"datatype": "except"
				  }
				}
			  },
			"FarthestFirst": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/clusterers/FarthestFirst.html",
				"params": {
				  "-S": {
					"default": 1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  }
				}
			  },
			"Canopy": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/clusterers/Canopy.html",
				"params": {
				  "-max-candidates": {
					"default": 100,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">",
					  "value": 1
					}
				  },
				  "-periodic-pruning": {
					"default": 10000,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<>",
					  "value": 0
					}
				  },
				  "-min-density": {
					"default": 2,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-t2": {
					"default": -1.0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-t1": {
					"default": -1.5,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-M": {
					"default": False,
					"datatype": "except"
				  },
				  "-S": {
					"default": 1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
				  "-output-debug-info": {
					"default": False,
					"datatype": "except"
				  },
				  "-do-not-check-capabilities": {
					"default": False,
					"datatype": "except"
				  }
				}
			  },
			"SimpleKMeans": {
				"reference_link": "https://javadoc.io/static/nz.ac.waikato.cms.weka/weka-stable/3.8.3/weka/clusterers/SimpleKMeans.html",
				"params": {
				  "-init": {
					"default": 0,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=",
					  "value": 3
					}
				  },
					"-C": {
					"default": False,
					"datatype": "except"
				  },
					"-max-candidates": {
					"default": 100,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
					"-periodic-pruning": {
					"default": 10000,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value":  0
					}
				  },
					"-min-density": {
					"default": 2,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
					"-t2": {
					"default": -1.0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
					"-t1": {
					"default": -1.5,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
					"-V": {
					"default": False,
					"datatype": "except"
				  },
					"-M": {
					"default": False,
					"datatype": "except"
				  },
					"-I": {
					"default": 1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
					"-O": {
					"default": False,
					"datatype": "except"
				  },
					"-fast": {
					"default": False,
					"datatype": "except"
				  },
					"-num-slots": {
					"default": 1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": ">=",
					  "value": 1
					}
				  },
					"-S": {
					"default": 10,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
					  "symbol": "<=>",
					  "value": 0
					}
				  },
					"-output-debug-info": {
					"default": False,
					"datatype": "except"
				  },
					"-do-not-check-capabilities": {
					"default": False,
					"datatype": "except"

				  }
				}
			  }
			}
	},
	"h2o": {
		"regression": {
			"LinearRegression": {
				"reference_link": "https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/glm.html#H2OGeneralizedLinearEstimator",
				"params": {
					"HGLM": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"auc_type": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"none",
							"macro_ovr",
							"weighted_ovr",
							"macro_ovo",
							"weighted_ovo"
						]
					},
					"balance_classes": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"beta_epsilon": {
						"default": 0.0001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"calc_like": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"cold_start": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"compute_p_values": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"early_stopping": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"family": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"gaussian",
							"binomial",
							"fractionalbinomial",
							"quasibinomial",
							"ordinal",
							"multinomial",
							"poisson",
							"gamma",
							"tweedie",
							"negativebinomial"
						]
					},
					"fold_assignment": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"random",
							"modulo",
							"stratified"
						]
					},
					"gradient_epsilon": {
						"default": -1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"ignore_const_cols": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"intercept": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"keep_cross_validation_fold_assignment": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"keep_cross_validation_models": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"keep_cross_validation_predictions": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"lambda_min_ratio": {
						"default": -1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"lambda_search": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"link": {
						"default": "family_default",
						"datatype": "str",
						"value": [
							"family_default",
							"identity",
							"logit",
							"log",
							"inverse",
							"tweedie",
							"ologit"
						]
					},
					"max_active_predictors": {
						"default": -1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": -1
						}
					},
					"max_after_balance_size": {
						"default": 5.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"max_confusion_matrix_size": {
						"default": 20,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0
						}
					},
					"max_iterations": {
						"default": -1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": "check",
							"value": 0
						}
					},
					"max_runtime_secs": {
						"default": 0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"missing_values_handling": {
						"default": "mean_imputation",
						"datatype": "str",
						"value": [
							"mean_imputation",
							"skip",
							"plug_values"
						]
					},
					"nlambdas": {
						"default": -1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0
						}
					},
					"non_negative": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"obj_reg": {
						"default": -1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "check",
							"value": 0.0
						}
					},
					"objective_epsilon": {
						"default": -1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"prior": {
						"default": -1,
						"datatype": "float",
						"compare_type": "check"
					},
					"remove_collinear_columns": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"score_each_iteration": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"score_iteration_interval": {
						"default": -1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0
						}
					},
					"seed": {
						"default": -1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0
						}
					},
					"solver": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"irlsm",
							"l_bfgs",
							"coordinate_descent_naive",
							"coordinate_descent",
							"gradient_descent_lh",
							"gradient_descent_sqerr"
						]
					},
					"standardize": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"stopping_metric": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"deviance",
							"logloss",
							"mse",
							"rmse",
							"mae",
							"rmsle",
							"auc",
							"aucpr",
							"lift_top_group",
							"misclassification",
							"mean_per_class_error",
							"custom",
							"custom_increasing"
						]
					},
					"stopping_rounds": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"stopping_tolerance": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0,
							1
						]
					},
					"theta": {
						"default": 1e-10,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"tweedie_link_power": {
						"default": 1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"tweedie_variance_power": {
						"default": 0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					}
				}
			},
			"GradientBoostingMachines": {
				"reference_link": "https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/gbm.html#H2OGradientBoostingEstimator",
				"params": {
					"auc_type": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"none",
							"macro_ovr",
							"weighted_ovr",
							"macro_ovo",
							"weighted_ovo"
						]
					},
					"balance_classes": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"build_tree_one_node": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"calibrate_model": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"categorical_encoding": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"enum",
							"one_hot_internal",
							"one_hot_explicit",
							"binary",
							"eigen",
							"label_encoder",
							"sort_by_response",
							"enum_limited"
						]
					},
					"check_constant_response": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"col_sample_rate": {
						"default": 1,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							1.0
						]
					},
					"col_sample_rate_change_per_level": {
						"default": 1,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							2.0
						]
					},
					"col_sample_rate_per_tree": {
						"default": 1,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							1.0
						]
					},
					"distribution": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"bernoulli",
							"quasibinomial",
							"multinomial",
							"gaussian",
							"poisson",
							"gamma",
							"tweedie",
							"laplace",
							"quantile",
							"huber",
							"custom"
						]
					},
					"fold_assignment": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"random",
							"modulo",
							"stratified"
						]
					},
					"gainslift_bins": {
						"default": -1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": -1
						}
					},
					"histogram_type": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"uniform_adaptive",
							"random",
							"quantiles_global",
							"round_robin"
						]
					},
					"huber_alpha": {
						"default": 0.9,
						"datatype": "float",
						"compare_type": "check"
					},
					"ignore_const_cols": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"keep_cross_validation_fold_assignment": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"keep_cross_validation_models": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"keep_cross_validation_predictions": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"learn_rate": {
						"default": 0.1,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							1.0
						]
					},
					"learn_rate_annealing": {
						"default": 1,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							1.0
						]
					},
					"max_abs_leafnode_pred": {
						"default": 1.7976,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"max_after_balance_size": {
						"default": 5.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"max_confusion_matrix_size": {
						"default": 20,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0
						}
					},
					"max_depth": {
						"default": 5,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"max_runtime_secs": {
						"default": 0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_rows": {
						"default": 10,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"min_split_improvement": {
						"default": 0.00001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"nbins": {
						"default": 20,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"nbins_cats": {
						"default": 1024,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"nbins_top_level": {
						"default": 1024,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": "check",
							"value": 0
						}
					},
					"ntrees": {
						"default": 50,
						"datatype": "int",
						"compare_type": "range",
						"range": [
							1,
							100000
						]
					},
					"pred_noise_bandwidth": {
						"default": 0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"quantile_alpha": {
						"default": 0.5,
						"datatype": "float",
						"compare_type": "check"
					},
					"r2_stopping": {
						"default": 1.7976,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0.0
						}
					},
					"sample_rate": {
						"default": 1,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							1.0
						]
					},
					"score_each_iteration": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"score_tree_interval": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"seed": {
						"default": -1,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0
						}
					},
					"stopping_metric": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"deviance",
							"logloss",
							"mse",
							"rmse",
							"mae",
							"rmsle",
							"auc",
							"aucpr",
							"lift_top_group",
							"misclassification",
							"mean_per_class_error",
							"custom",
							"custom_increasing"
						]
					},
					"stopping_rounds": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"stopping_tolerance": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0.0,
							1.0
						]
					},
					"tweedie_power": {
						"default": 1.5,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							1,
							2
						]
					}
				}
			},
			"RandomForest": {
				"reference_link": "https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/random_forest.html",
				"params": {
					"auc_type": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"none",
							"macro_ovr",
							"weighted_ovr",
							"macro_ovo",
							"weighted_ovo"
						]
					},
					"balance_classes": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"binomial_double_trees": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"build_tree_one_node": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"calibrate_model": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"categorical_encoding": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"enum",
							"one_hot_internal",
							"one_hot_explicit",
							"binary",
							"eigen",
							"label_encoder",
							"sort_by_response",
							"enum_limited"
						]
					},
					"check_constant_response": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"col_sample_rate_change_per_level": {
						"default": 1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0
						},
						"range": [
							0.0,
							2.0
						]
					},
					"col_sample_rate_per_tree": {
						"default": 1,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0,
							6
						]
					},
					"distribution": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"bernoulli",
							"multinomial",
							"gaussian",
							"poisson",
							"gamma",
							"tweedie"
						]
					},
					"fold_assignment": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"random",
							"modulo",
							"stratified"
						]
					},
					"histogram_type": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"uniform_adaptive",
							"random",
							"quantiles_global",
							"round_robin"
						]
					},
					"keep_cross_validation_fold_assignment": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"keep_cross_validation_models": {
						"default": True,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"keep_cross_validation_predictions": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"max_after_balance_size": {
						"default": 1.0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"max_confusion_matrix_size": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"max_depth": {
						"default": 0,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"max_runtime_secs": {
						"default": 0,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"min_rows": {
						"default": 1,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"min_split_improvement": {
						"default": 0.00001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 0.0
						}
					},
					"mtries": {
						"default": -1,
						"datatype": "hybrid",
						"condition": {
							"symbol": ">=",
							"value": 1
						},
						"range": [
							-2.0,
							-1.0
						]
					},
					"nbins": {
						"default": 2,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"nbins_cats": {
						"default": 1024,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"nbins_top_level": {
						"default": 1024,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"ntrees": {
						"default": 50,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">",
							"value": 1
						}
					},
					"r2_stopping": {
						"default": -1.7976,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": "<=>",
							"value": 0
						}
					},
					"sample_rate": {
						"default": 0.632,
						"datatype": "float",
						"compare_type": "range",
						"range": [
							0,
							1
						]
					},
					"score_each_iteration": {
						"default": False,
						"datatype": "str",
						"value": [
							True,
							False
						]
					},
					"score_tree_interval": {
						"default": 10,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"stopping_metric": {
						"default": "auto",
						"datatype": "str",
						"value": [
							"auto",
							"deviance",
							"logloss",
							"mse",
							"rmse",
							"mae",
							"rmsle",
							"auc",
							"aucpr",
							"lift_top_group",
							"misclassification",
							"mean_per_class_error",
							"custom",
							"custom_increasing"
						]
					},
					"stopping_rounds": {
						"default": 10,
						"datatype": "int",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0
						}
					},
					"stopping_tolerance": {
						"default": 0.001,
						"datatype": "float",
						"compare_type": "condition",
						"condition": {
							"symbol": ">=",
							"value": 0.0
						}
					},
					"class_sampling_factors": {
						"default": None,
						"datatype": "except"
					},
					"sample_rate_per_class": {
						"default": None,
						"datatype": "except"
					}
				}
			}
		},
		"classification": {
		"NaiveBayesBinomial": {
			"reference_link": "https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/naive_bayes.html",
			"params": {
				"auc_type": {
					"default": "auto",
					"datatype": "str",
					"value": [
						"auto",
						"none",
						"macro_ovr",
						"weighted_ovr",
						"macro_ovo",
						"weighted_ovo"
					]
				},
				"balance_classes": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"compute_metrics": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"eps_prob": {
					"default": 30,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 2.0
					}
				},
				"eps_sdev": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0.0
					}
				},
				"fold_assignment": {
					"default": "auto",
					"datatype": "str",
					"value": [
						"auto",
						"random",
						"modulo",
						"stratified"
					]
				},
				"gainslift_bins": {
					"default": -1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0
					}
				},
				"ignore_const_cols": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"keep_cross_validation_fold_assignment": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"keep_cross_validation_models": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"keep_cross_validation_predictions": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"laplace": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0.0
					}
				},
				"max_after_balance_size": {
					"default": 5,
					"datatype": "float",
					"compare_type": ">="
				},
				"max_confusion_matrix_size": {
					"default": 0,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0
					}
				},
				"max_runtime_secs": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0.0
					}
				},
				"min_prob": {
					"default": 0.001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0.0
					}
				},
				"min_sdev": {
					"default": 0.001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0.0
					}
				},
				"score_each_iteration": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"seed": {
					"default": -1,
					"datatype": "except"
				}
			}
		},
		"DeepLearning": {
			"reference_link": "https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/deeplearning.html#H2ODeepLearningEstimator",
			"params": {
				"activation": {
					"default": "rectifier",
					"datatype": "str",
					"value": [
						"tanh",
						"tanh_with_dropout",
						"rectifier",
						"rectifier_with_dropout",
						"maxout",
						"maxout_with_dropout"
					]
				},
				"adaptive_rate": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"auc_type": {
					"default": "auto",
					"datatype": "str",
					"value": [
						"auto",
						"none",
						"macro_ovr",
						"weighted_ovr",
						"macro_ovo",
						"weighted_ovo"
					]
				},
				"autoencoder": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"average_activation": {
					"default": 0.0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"balance_classes": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"categorical_encoding": {
					"default": "auto",
					"datatype": "str",
					"value": [
						"auto",
						"enum",
						"one_hot_internal",
						"one_hot_explicit",
						"binary",
						"eigen",
						"label_encoder",
						"sort_by_response",
						"enum_limited"
					]
				},
				"classification_stop": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"col_major": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"diagnostics": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"distribution": {
					"default": "auto",
					"datatype": "str",
					"value": [
						"auto",
						"bernoulli",
						"multinomial",
						"gaussian",
						"poisson",
						"gamma",
						"tweedie",
						"laplace",
						"quantile",
						"huber"
					]
				},
				"elastic_averaging": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"elastic_averaging_moving_rate": {
					"default": 0.9,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"elastic_averaging_regularization": {
					"default": 0.001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"epochs": {
					"default": 10,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"epsilon": {
					"default": 1e-8,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">",
						"value": 0.0
					}
				},
				"export_weights_and_biases": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"fast_mode": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"fold_assignment": {
					"default": "auto",
					"datatype": "str",
					"value": [
						"auto",
						"random",
						"modulo",
						"stratified"
					]
				},
				"force_load_balance": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"huber_alpha": {
					"default": 0.9,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"ignore_const_cols": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"initial_weight_distribution": {
					"default": "uniform_adaptive",
					"datatype": "str",
					"value": [
						"uniform_adaptive",
						"uniform",
						"normal"
					]
				},
				"initial_weight_scale": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"input_dropout_ratio": {
					"default": 0,
					"datatype": "float",
					"compare_type": "range",
					"range": [
						0,
						1
					]
				},
				"keep_cross_validation_fold_assignment": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"keep_cross_validation_models": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"keep_cross_validation_predictions": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"l1": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0.0
					}
				},
				"l2": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0.0
					}
				},
				"loss": {
					"default": "automatic",
					"datatype": "str",
					"value": [
						"automatic",
						"cross_entropy",
						"quadratic",
						"huber",
						"absolute",
						"quantile"
					]
				},
				"max_after_balance_size": {
					"default": 5.0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"max_categorical_features": {
					"default": 2147483647,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": ">",
						"value": 0
					}
				},
				"max_confusion_matrix_size": {
					"default": 20,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0
					}
				},
				"max_runtime_secs": {
					"default": 0.0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0.0
					}
				},
				"max_w2": {
					"default": 3.4028235e38,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": ">",
						"value": 0.0
					}
				},
				"mini_batch_size": {
					"default": 1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": ">",
						"value": 0
					}
				},
				"missing_values_handling": {
					"default": "mean_imputation",
					"datatype": "str",
					"value": [
						"mean_imputation",
						"skip"
					]
				},
				"momentum_ramp": {
					"default": 1000000,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"momentum_stable": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"momentum_start": {
					"default": 0,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"nesterov_accelerated_gradient": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"overwrite_with_best_model": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"quantile_alpha": {
					"default": 0.5,
					"datatype": "float",
					"compare_type": "range",
					"range": [
						0,
						1
					]
				},
				"quiet_mode": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"rate": {
					"default": 0.005,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"rate_annealing": {
					"default": 0.000001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"rate_decay": {
					"default": 1,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"regression_stop": {
					"default": 0.000001,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"replicate_training_data": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"reproducible": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"rho": {
					"default": 0.99,
					"datatype": "float",
					"compare_type": "range",
					"range": [
						0.0,
						0.99
					]
				},
				"score_duty_cycle": {
					"default": 0.1,
					"datatype": "float",
					"compare_type": "range",
					"range": [
						0,
						1
					]
				},
				"score_each_iteration": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"score_interval": {
					"default": 5,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"score_training_samples": {
					"default": 10000,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0
					}
				},
				"score_validation_samples": {
					"default": 0,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0
					}
				},
				"score_validation_sampling": {
					"default": "uniform",
					"datatype": "str",
					"value": [
						"uniform",
						"stratified"
					]
				},
				"seed": {
					"default": -1,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0
					}
				},
				"shuffle_training_data": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"single_node_mode": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"sparse": {
					"default": False,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"sparsity_beta": {
					"default": 0,
					"datatype": "float",
					"compare_type": "check"
				},
				"standardize": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"stopping_metric": {
					"default": "auto",
					"datatype": "str",
					"value": [
						"auto",
						"deviance",
						"logloss",
						"mse",
						"rmse",
						"mae",
						"rmsle",
						"auc",
						"aucpr",
						"lift_top_group",
						"misclassification",
						"mean_per_class_error",
						"custom",
						"custom_increasing"
					]
				},
				"stopping_rounds": {
					"default": 5,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": ">=",
						"value": 0
					}
				},
				"stopping_tolerance": {
					"default": 0,
					"datatype": "float",
					"compare_type": "range",
					"range": [
						0,
						1
					]
				},
				"target_ratio_comm_to_comp": {
					"default": 0.05,
					"datatype": "float",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0.0
					}
				},
				"train_samples_per_iteration": {
					"default": -2,
					"datatype": "int",
					"compare_type": "condition",
					"condition": {
						"symbol": "<=>",
						"value": 0
					}
				},
				"tweedie_power": {
					"default": 1.5,
					"datatype": "float",
					"compare_type": "range",
					"range": [
						1,
						2
					]
				},
				"use_all_factor_levels": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				},
				"variable_importances": {
					"default": True,
					"datatype": "str",
					"value": [
						True,
						False
					]
				}
			}
		}
	},
		"clustering":{
	"KMeansClustering": {"reference_link": "https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/kmeans.html",
		"params":{"categorical_encoding": {
			"default": "auto",
			"datatype": "str",
			"value": [
				"auto",
				"enum",
				"one_hot_internal",
				"one_hot_explicit",
				"binary",
				"eigen",
				"label_encoder",
				"sort_by_response",
				"enum_limited"
			]
		},
		"cluster_size_constraints": {
			"default": None,
			"datatype": "except"
		},
		"estimate_k": {
			"default": False,
			"datatype": "str",
			"value": [
				True,
				False
			]
		},
		"export_checkpoints_dir": {
			"default": None,
			"datatype": "except"
		},
		"fold_assignment": {
			"default": "auto",
			"datatype": "str",
			"value": [
				"auto",
				"random",
				"modulo",
				"stratified"
			]
		},
		"fold_column": {
			"default": None,
			"datatype": "except"
		},
		"ignore_const_cols": {
			"default": True,
			"datatype": "str",
			"value": [
				True,
				False
			]
		},
		"ignored_columns": {
			"default": None,
			"datatype": "except"
		},
		"init": {
			"default": "furthest",
			"datatype": "str",
			"value": [
				"random",
				"plus_plus",
				"furthest",
				"user"
			]
		},
		"k": {
			"default": 1,
			"datatype": "int",
			"compare_type": "range",
			"range": [
				1,
				1e7
			]
		},
		"keep_cross_validation_fold_assignment": {
			"default": False,
			"datatype": "str",
			"value": [
				True,
				False
			]
		},
		"keep_cross_validation_models": {
			"default": True,
			"datatype": "str",
			"value": [
				True,
				False
			]
		},
		"keep_cross_validation_predictions": {
			"default": False,
			"datatype": "str",
			"value": [
				True,
				False
			]
		},
		"max_iterations": {
			"default": 10,
			"datatype": "int",
			"compare_type": "range",
			"range": [
				1,
				0.000001
			]
		},
		"max_runtime_secs": {
			"default": 0,
			"datatype": "float",
			"compare_type": "condition",
			"condition": {
				"symbol": ">=",
				"value": 0.0
			}
		},
		"nfolds": {
			"default": 0,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": ">",
				"value": 1
			}
		},
		"score_each_iteration": {
			"default": False,
			"datatype": "str",
			"value": [
				True,
				False
			]
		},
		"seed": {
			"default": -1,
			"datatype": "int",
			"compare_type": "condition",
			"condition": {
				"symbol": "<=>",
				"value": 0
			}
		},
		"standardize": {
			"default": True,
			"datatype": "str",
			"value": [
				True,
				False
			]
		},
		"user_points": {
			"default": None,
			"datatype": "except"
		}
	}
}
		}
	}
	}