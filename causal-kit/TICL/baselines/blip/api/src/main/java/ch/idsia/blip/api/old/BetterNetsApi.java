package ch.idsia.blip.api.old;


import ch.idsia.blip.api.common.NetToGraphApi;
import ch.idsia.blip.core.utils.other.BetterNets;

import java.util.logging.Logger;

public class BetterNetsApi extends NetToGraphApi {

    private static final Logger log = Logger.getLogger(
            NetToGraphApi.class.getName());

    public BetterNetsApi() {
        ntg = new BetterNets();
    }

    public static void main(String[] args) {
        defaultMain(args, new BetterNetsApi());
    }
}
