import sys
import torch

from lassonet import LassoNetClassifier
from lassonet.interfaces import HistoryItem


class LassoNetSAMInputClassifier(LassoNetClassifier):
    # LassoNet binary classification with SAM-style input perturbation training.
    

    def __init__(
        self,
        adv_rho=0.0,
        adv_alpha=0.0,
        adv_delta=1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adv_rho = adv_rho
        self.adv_alpha = adv_alpha
        self.adv_delta = adv_delta

    def _train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        batch_size,
        epochs,
        lambda_,
        optimizer,
        return_state_dict,
        patience=None,
    ):
        model = self.model
        # Validation objective: cross-entropy plus regularization terms.
        def validation_obj():
            with torch.no_grad():
                return (
                    self.criterion(model(X_val), y_val).item()
                    + lambda_ * model.l1_regularization_skip().item()
                    + self.gamma * model.l2_regularization().item()
                    + self.gamma_skip * model.l2_regularization_skip().item()
                )

        best_val_obj = validation_obj()
        epochs_since_best_val_obj = 0
        if self.backtrack:
            best_state_dict = self.model.cpu_state_dict()
            real_best_val_obj = best_val_obj
            real_loss = float("nan")

        n_iters = 0

        n_train = len(X_train)
        if batch_size is None:
            batch_size = n_train
            randperm = torch.arange
        else:
            randperm = torch.randperm
        batch_size = min(batch_size, n_train)
        # Enable input perturbation only when both rho and alpha are positive.
        adv_enabled = self.adv_rho > 0 and self.adv_alpha > 0

        for epoch in range(epochs):
            indices = randperm(n_train)
            model.train()
            loss = 0
            for i in range(n_train // batch_size):
                batch = indices[i * batch_size : (i + 1) * batch_size]

                def closure():
                    nonlocal loss
                    optimizer.zero_grad()

                    Xb = X_train[batch]
                    yb = y_train[batch]

                    if adv_enabled:
                        # Enable input gradients to compute ∇x L_clean.
                        Xb = Xb.detach().requires_grad_(True)
                    # Cross-entropy on clean samples.
                    crit_clean = self.criterion(model(Xb), yb)
                    crit = crit_clean

                    if adv_enabled:
                        # One-step SAM-style perturbation with L2 normalization.
                        g = torch.autograd.grad(crit_clean, Xb, retain_graph=True)[0]
                        g_norm = torch.norm(g, p=2)
                        r = self.adv_rho * g / (g_norm + self.adv_delta)
                        # Generate adversarial samples and run the forward pass.
                        Xb_adv = (Xb + r).detach()
                        crit_adv = self.criterion(model(Xb_adv), yb)
                        # Align with adv.py: mix clean and adversarial losses, both with gradients.
                        crit = (1 - self.adv_alpha) * crit_clean + self.adv_alpha * crit_adv

                    ans = (
                        crit
                        + self.gamma * model.l2_regularization()
                        + self.gamma_skip * model.l2_regularization_skip()
                    )

                    if not torch.isfinite(ans):
                        print(f"Loss is {ans}", file=sys.stderr)
                        print("Did you normalize input?", file=sys.stderr)
                        print("Loss::", crit.item())
                        print("l2_regularization:", model.l2_regularization())
                        print("l2_regularization_skip:", model.l2_regularization_skip())
                        assert False

                    ans.backward()
                    loss += ans.item() * batch_size / n_train
                    return ans

                optimizer.step(closure)
                # Hierarchical proximal update for the LassoNet layer.
                model.prox(
                    lambda_=lambda_ * optimizer.param_groups[0]["lr"],
                    M=self.M,
                )

            if epoch == 0:
                real_loss = loss

            model.eval()
            val_obj = validation_obj()
            if val_obj < self.tol * best_val_obj:
                best_val_obj = val_obj
                epochs_since_best_val_obj = 0
            else:
                epochs_since_best_val_obj += 1

            if self.backtrack and val_obj < real_best_val_obj:
                best_state_dict = self.model.cpu_state_dict()
                real_best_val_obj = val_obj
                real_loss = loss
                n_iters = epoch + 1

            if patience is not None and epochs_since_best_val_obj == patience:
                break

        if self.backtrack:
            self.model.load_state_dict(best_state_dict)
            val_obj = real_best_val_obj
            loss = real_loss
        else:
            n_iters = epoch + 1

        with torch.no_grad():
            reg = self.model.l1_regularization_skip().item()
            l2_regularization = self.model.l2_regularization()
            l2_regularization_skip = self.model.l2_regularization_skip()

        return HistoryItem(
            lambda_=lambda_,
            state_dict=self.model.cpu_state_dict() if return_state_dict else None,
            objective=loss + lambda_ * reg,
            loss=loss,
            val_objective=val_obj,
            val_loss=val_obj - lambda_ * reg,
            regularization=reg,
            l2_regularization=l2_regularization,
            l2_regularization_skip=l2_regularization_skip,
            selected=self.model.input_mask().cpu(),
            n_iters=n_iters,
        )
