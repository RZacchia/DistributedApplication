using System.Security.Principal;
using BookRent.Orchestrator.Api.Requests;
using BookRent.Orchestrator.Clients;

namespace BookRent.Orchestrator.Services.Interfaces;

public class IdentityService : IIdentityService
{
    private readonly IdentityClient _identityClient;
    
    public IdentityService(IdentityClient identityClient)
    {
        _identityClient = identityClient;
    }


    public async Task<string> LoginRequest(AuthentificationRequest request)
    {
        var login = new LoginRequest
        {
            
        };
        return await _identityClient.LoginAsync(login);
    }
}