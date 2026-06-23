.PHONY: verify quick evidence report package

verify:
	bash scripts/verify_artifact.sh

quick:
	tools/tilemem verify --quick

evidence:
	tools/tilemem evidence verify

report:
	bash scripts/reproduce_ablation.sh

package:
	bash scripts/package_release.sh
